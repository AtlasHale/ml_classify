import wandb


api = wandb.Api()
run = api.run('mbari/tf_five_learning/runs/xmu7fgyn')
data = run.history(pandas=True)
data.to_excel('/Users/chale/Desktop/lc.xlsx', engine='xlsxwriter')
