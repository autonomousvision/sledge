CHALLENGE=sledge_reactive_agents 

python $SLEDGE_DEVKIT_ROOT/sledge/script/run_simulation.py \
+simulation=$CHALLENGE \
planner=pdm_closed_planner \
observation=sledge_agents_observation \
scenario_builder=nuplan 