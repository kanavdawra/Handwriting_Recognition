def get_random_data_by_percentage(dataframe, percentage):
    count = dataframe.shape[0]
    desired_count = int(round(count * percentage / 100, 0))
    dataframe = dataframe.sample(n=desired_count)
    dataframe.reset_index(drop=True,inplace=True)
    return dataframe
