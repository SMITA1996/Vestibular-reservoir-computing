# Vestibular-reservoir-computing

Vestibular reservoir computing (VRC) model for the  prediction of chaotic dynamics by Smita Deb and Shirin Panahi.

This is the repository for our preprint titled "Vestibular reservoir computing". This research focuses on using a vestibular reservoir computing model to predict chaotic time series in different chaotic systems using different network topologies, such as coupled and uncoupled reservoirs.

# User instructions
Compilation of the given codes requires MATLAB version R2020b. Download all the .m files and keep them in the same folder.

# Codes and results
Compile the scripts rc_VRC_lor.m and rc_VRC_FC.m to obtain predictions of chaotic dynamics for the Lorenz and Food Chain (FC) systems, respectively.

The codes first generate the corresponding system data and then perform training, validation, and testing using the VRC framework. They also provide visualization of the training, validation, and testing phases, along with computation of both short-term and long-term prediction statistics.

The script MF_MC_VRC.m is used to compute the memory function and memory capacity of the VRC using i.i.d. inputs.

For data generation, compile func_generate_data_lorenz.m for the Lorenz system and foodchain_sim.m for the chaotic Food Chain system.




