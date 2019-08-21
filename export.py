import wandb


api = wandb.Api()
run = api.run('mbari/inception_training/runs/erijpv2c')
data = run.history(pandas=True)
data.to_excel('/Users/ereyes/Desktop/what.xlsx', engine='xlsxwriter')
