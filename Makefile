all:

clean:
	@echo 'Are you sure? Sleeping for 15 seconds in case you changed your mind!' | cowsay
# 	@sleep 15
	rm -rf params/*
	rm -rf tb_logs/*
	rm -rf attacker_payoff.npy
	rm -rf defender_payoff.npy
	rm -rf log.log
