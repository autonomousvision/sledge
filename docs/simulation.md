# Simulation

This section provides instructions for simulating scenarios in SLEDGE.

### Simple Simulation

For the v0.1 release, you can simulate simple 64mx64m patches. You first need to train a diffusion model and run scenario caching, as described in `docs/diffusion.md`.
Consequently, you can simulate the scenarios by running.
```bash
cd $SLEDGE_DEVKIT_ROOT/scripts/simulation/
bash simple_simulation.sh
``` 
By default, we simulate the [PDM-Closed](https://arxiv.org/abs/2306.07962) planner for 15 seconds. The experiment folder can be found in `$SLEDGE_EXP_ROOT/exp`. Further simulation modes and configurations will follow in future updates.

### Visualization

The simulated scenarios can be visualized with SledgeBoard. Simply run:
```bash 
python $SLEDGE_DEVKIT_ROOT/sledge/script/run_sledgeboard.py
```
Open the `.nuboard` file in the experiment folder, view simulations, and render videos of scenarios. Note that SledgeBoard is primarily of a skin of nuBoard, with small adaptations to view synthetic scenarios in SLEDGE.
