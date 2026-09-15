from pathlib import Path
path = Path(__file__).with_name('quick_shanghai_counterparts.py')
src = path.read_text()
src = src.replace("for mode in ['post','pre']:", "for mode in ['pre']:")
src = src.replace("out=H/f'runs_shanghai_{mode}_supervised_seed1341'", "out=H/f'runs_shanghai_{mode}_supervised_3seeds'")
src = src.replace("seeds = [1341]", "seeds = [1341,1535,1631]")
src = src.replace("nDataSet = 1", "nDataSet = 3")
exec(compile(src, str(path), 'exec'))
