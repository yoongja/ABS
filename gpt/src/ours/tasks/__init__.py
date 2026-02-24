def get_task(name):
    tasks = ['game24', 'commonsenseqa', 'strategyqa', 'aqua', 'gsm8k', 'date']
    # if name in tasks:
    #     from tot.tasks.general_task import GeneralTask
    #     return GeneralTask(name)
    if name == 'game24':
        from tot.tasks.game24_ours import Game24Task
        return Game24Task()
    elif name == 'commonsenseqa':
        from tot.tasks.commonsenseqa import CommonsenseqaTask
        return CommonsenseqaTask()
    elif name == 'gsm8k':
        from tot.tasks.gsm8k import GSM8KTask
        return GSM8KTask()
    elif name == 'aqua':
        from tot.tasks.aqua import AQUATask
        return AQUATask()
    elif name == 'strategyqa':
        from tot.tasks.strategyqa import StrategyqaTask
        return StrategyqaTask()
    elif name == 'date_understanding':
        from tot.tasks.date import DateUnderstandingTask
        return DateUnderstandingTask()
        
    else:
        raise NotImplementedError