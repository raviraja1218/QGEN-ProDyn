# 03_md_simulations/run_md.py
import os, csv, json, sys, time
from openmm import Platform, LangevinIntegrator, unit
from openmm.app import (
    PDBFile, ForceField, Simulation, Modeller,
    DCDReporter, StateDataReporter,
    NoCutoff, CutoffNonPeriodic, CutoffPeriodic, PME, HBonds
)
from openmm.openmm import CustomExternalForce

ROOT = os.path.dirname(__file__)

def _load_json(path):
    with open(path) as f:
        return json.load(f)

PARAMS = _load_json(os.path.join(ROOT, "params.json"))
RUNPLAN = [row for row in csv.DictReader(open(os.path.join(ROOT, "run_plan.csv")))]

OUT_TOP  = os.path.join(ROOT, "topologies")
OUT_TRAJ = os.path.join(ROOT, "trajectories")
OUT_LOGS = os.path.join(ROOT, "logs")
os.makedirs(OUT_TOP, exist_ok=True)
os.makedirs(OUT_TRAJ, exist_ok=True)
os.makedirs(OUT_LOGS, exist_ok=True)

def _nb_method_from_name(name: str):
    lut = {"NoCutoff": NoCutoff, "CutoffNonPeriodic": CutoffNonPeriodic,
           "CutoffPeriodic": CutoffPeriodic, "PME": PME}
    if name not in lut:
        raise ValueError(f"Unknown nonbonded_method '{name}'. Use {list(lut)}")
    return lut[name]

def add_positional_restraints(system, topology, reference_positions, k_kcal_per_A2=2.0, heavy_only=True):
    k = k_kcal_per_A2 * 418.4 * (unit.kilojoule_per_mole / (unit.nanometer**2))
    force = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    force.addGlobalParameter("k", k)

    pos_nm = [p.value_in_unit(unit.nanometer) for p in reference_positions]
    for idx, atom in enumerate(topology.atoms()):
        if heavy_only and atom.element is not None and atom.element.symbol == "H":
            continue
        x0, y0, z0 = pos_nm[idx]
        force.addParticle(idx, [x0, y0, z0])

    system.addForce(force)
    return system

def make_integrator(dt_fs: float, T_K: float, gamma_ps: float):
    return LangevinIntegrator(T_K*unit.kelvin, gamma_ps/unit.picosecond, dt_fs*unit.femtoseconds)

def build_system_and_sim(pdb_in, nonbonded_name, platform_name, pH, constraints_name):
    print("[run_md] Loading PDB:", os.path.basename(pdb_in)); sys.stdout.flush()
    pdb = PDBFile(pdb_in)
    modeller = Modeller(pdb.topology, pdb.positions)

    ff = ForceField("amber14-all.xml", "implicit/gbn2.xml")

    print("[run_md] Adding hydrogens at pH", pH); sys.stdout.flush()
    modeller.addHydrogens(forcefield=ff, pH=float(pH))

    nb_method = _nb_method_from_name(nonbonded_name)
    if nb_method in (CutoffPeriodic, PME):
        print(f"[run_md] Requested '{nonbonded_name}' with implicit solvent; forcing 'NoCutoff'.")
        nb_method = NoCutoff

    constraints = HBonds if str(constraints_name).lower() == "hbonds" else None

    print("[run_md] Creating system (implicit solvent, NoCutoff)…"); sys.stdout.flush()
    system = ff.createSystem(modeller.topology, nonbondedMethod=nb_method, constraints=constraints)

    platform = Platform.getPlatformByName(platform_name)
    return modeller, system, platform

def production_with_safety(modeller, system, platform, outfile_dcd, outfile_log, outfile_top, seed):
    T_K          = float(PARAMS.get("temperature_K", 300.0))
    gamma_ps     = float(PARAMS.get("friction_ps", 1.0))
    report_every = int(PARAMS.get("report_every", 200))
    dcd_stride   = int(PARAMS.get("dcd_stride", 200))

    small_dt     = float(PARAMS.get("small_equil_dt_fs", 0.5))
    small_steps  = int(PARAMS.get("small_equil_steps", 3000))
    restr_k      = float(PARAMS.get("equil_restraints_k_kcal_A2", 5.0))

    dt_fs        = float(PARAMS.get("timestep_fs", 1.0))
    equil_steps  = int(PARAMS.get("equil_steps", 4000))
    nsteps       = int(PARAMS.get("nsteps", 50000))

    # Restrained small-dt equilibration
    print(f"[run_md] Minimization…"); sys.stdout.flush()
    system_eq = add_positional_restraints(system, modeller.topology, modeller.positions, k_kcal_per_A2=restr_k, heavy_only=True)
    integ_eq  = make_integrator(small_dt, T_K, gamma_ps)
    sim_eq    = Simulation(modeller.topology, system_eq, integ_eq, platform)
    sim_eq.context.setPositions(modeller.positions)
    sim_eq.context.setVelocitiesToTemperature(T_K*unit.kelvin, int(seed))

    try:
        sim_eq.minimizeEnergy(maxIterations=500)
    except Exception as e:
        print(f"[run_md] Warning: minimizeEnergy raised {e}. Continuing…"); sys.stdout.flush()

    print(f"[run_md] Restrained equilibration: {small_steps} steps @ {small_dt} fs…"); sys.stdout.flush()
    sim_eq.step(small_steps)

    stabilized_positions = sim_eq.context.getState(getPositions=True).getPositions()

    # Unrestrained equilibration
    print(f"[run_md] Unrestrained equilibration: {equil_steps} steps @ {dt_fs} fs…"); sys.stdout.flush()
    integ_eq2 = make_integrator(dt_fs, T_K, gamma_ps)
    sim_eq2   = Simulation(modeller.topology, system, integ_eq2, platform)
    sim_eq2.context.setPositions(stabilized_positions)
    sim_eq2.context.setVelocitiesToTemperature(T_K*unit.kelvin, int(seed))
    try:
        sim_eq2.minimizeEnergy(maxIterations=200)
    except Exception as e:
        print(f"[run_md] Warning: second minimizeEnergy raised {e}. Continuing…"); sys.stdout.flush()
    sim_eq2.step(equil_steps)

    preprod_positions = sim_eq2.context.getState(getPositions=True).getPositions()

    # Production
    print(f"[run_md] Production: {nsteps} steps @ {dt_fs} fs…"); sys.stdout.flush()
    integ_prod = make_integrator(dt_fs, T_K, gamma_ps)
    sim = Simulation(modeller.topology, system, integ_prod, platform)
    sim.context.setPositions(preprod_positions)
    sim.context.setVelocitiesToTemperature(T_K*unit.kelvin, int(seed))
    sim.reporters.append(StateDataReporter(outfile_log, report_every,
                                           step=True, time=True, potentialEnergy=True,
                                           totalEnergy=True, temperature=True, speed=True, separator=","))
    sim.reporters.append(DCDReporter(outfile_dcd, dcd_stride))

    try:
        sim.minimizeEnergy(maxIterations=200)
    except Exception as e:
        print(f"[run_md] Warning: production minimizeEnergy raised {e}."); sys.stdout.flush()

    try:
        sim.step(nsteps)
    except Exception as e:
        print(f"[run_md] ⚠ NaN during production at dt={dt_fs} fs, retrying at 0.5 fs… ({e})"); sys.stdout.flush()
        integ_prod_retry = make_integrator(0.5, T_K, gamma_ps)
        sim_retry = Simulation(modeller.topology, system, integ_prod_retry, platform)
        sim_retry.context.setPositions(preprod_positions)
        sim_retry.context.setVelocitiesToTemperature(T_K*unit.kelvin, int(seed))
        sim_retry.reporters.extend(sim.reporters)
        sim_retry.minimizeEnergy(maxIterations=200)
        sim_retry.step(nsteps)
        sim = sim_retry

    with open(outfile_top, "w") as f:
        PDBFile.writeFile(modeller.topology,
                          sim.context.getState(getPositions=True).getPositions(),
                          f, keepIds=True)
    print("[run_md] ✅ Production complete."); sys.stdout.flush()

def run_one(rep, group, seed, pdb_in, topo_out, dcd_out, log_out):
    print(f"[run_md] ▶ Running rep={rep} group={group} seed={seed} pdb={os.path.basename(pdb_in)}"); sys.stdout.flush()
    modeller, system, platform = build_system_and_sim(
        pdb_in=pdb_in,
        nonbonded_name=PARAMS.get("nonbonded_method", "NoCutoff"),
        platform_name=PARAMS.get("platform", "CPU"),
        pH=PARAMS.get("ph", 7.4),
        constraints_name=PARAMS.get("constraints", "HBonds"),
    )
    production_with_safety(modeller, system, platform, dcd_out, log_out, topo_out, seed)
    print(f"[run_md] ✅ Done -> {os.path.basename(dcd_out)}"); sys.stdout.flush()

if __name__ == "__main__":
    for row in RUNPLAN:
        rep   = int(row["replicate"])
        group = row["group"]
        seed  = int(row["seed"])
        pdb_in   = os.path.join(ROOT, "initial_states_fixed", f"{group}_init_{rep}.pdb")
        topo_out = os.path.join(OUT_TOP,  f"{group}_{rep}_topology.pdb")
        dcd_out  = os.path.join(OUT_TRAJ, f"{group}_{rep}.dcd")
        log_out  = os.path.join(OUT_LOGS, f"{group}_{rep}.log")
        run_one(rep, group, seed, pdb_in, topo_out, dcd_out, log_out)
