bg_color white
set ray_opaque_background, off
load \\wsl.localhost\\Ubuntu\\home\\raviraja\\QGEN-ProDyn\\01_target_prep\\6OIM_clean_H.pdb
hide everything
show cartoon
color grey80, all
select pocket, resi 60-74
show surface, pocket
set transparency, 0.35, pocket
color yellow, pocket
orient pocket
ray 2000,1500
png \\wsl.localhost\\Ubuntu\\home\\raviraja\\QGEN-ProDyn\\12_manuscript_assets\\main_figures\\figure_1b_kras_structure.png, dpi=300
quit
