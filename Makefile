all:

clean:
	@echo 'Are you sure? Sleeping for 15 seconds in case you changed your mind!' | cowsay
 	@sleep 15
	rm -rf params/*
	rm -rf tb_logs/*
	rm -rf *.npy
	rm -rf *.log
