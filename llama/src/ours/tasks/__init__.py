def get_task(name):
    tasks = ['commonsenseqa', 'strategyqa', 'aqua', 'gsm8k', 'date']
    if name == 'commonsenseqa':
        from src.ours.tasks.commonsenseqa import CommonsenseqaTask
        return CommonsenseqaTask()
    elif name == 'gsm8k':
        from src.ours.tasks.gsm8k import GSM8KTask
        return GSM8KTask()
    elif name == 'aqua':
        from src.ours.tasks.aqua import AQUATask
        return AQUATask()
    elif name == 'strategyqa':
        from src.ours.tasks.strategyqa import StrategyqaTask
        return StrategyqaTask()
    elif name == 'date_understanding':
        from src.ours.tasks.date import DateUnderstandingTask
        return DateUnderstandingTask()
    else:
        raise NotImplementedError
