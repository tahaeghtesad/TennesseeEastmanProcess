all:
	@echo 'Usage:'
	@echo 'make [command]'
	@printf '\n'
	@echo 'Commands:'
	@printf '\t[clean]: Cleans the whole project.\n'
	@printf '\t[run]: Run with parameters in run.srun.py.\n'
	@printf '\t[stop]: Stop execution.\n'
	@printf '\t[count]: Number of running jobs on Sabine.\n'
	@printf '\t[list]: List running jobs on Sabine.\n'
	@printf '\t[watch]: Watch running jobs on Sabine.\n'

clean:
	@echo 'Are you sure? Sleeping for 15 seconds in case you changed your mind!' | cowsay
	@sleep 15
	rm -rf runs
	rm -rf slurm*

run:
		#module load Anaconda3/python-3.6
		python run.srun.py

stop:
		scancel -n TEP

count:
		@echo Total Submitted: $(squeue | grep TEP | wc -l)
		@echo Running $(squeue | grep TEP | grep R | wc -l)
		@echo Waiting for resources $(squeue | grep TEP | grep PD | wc -l)

list:
		squeue -l | grep TEP

watch:
		watch -n 10 "squeue -l | grep TEP | sort -k1 -n"

