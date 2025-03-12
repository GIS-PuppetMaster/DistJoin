import os
# data_path = 'D:/PycharmProjects/AQP/datasets/job/'
data_path = '/home/zkx/AQP/datasets/job/'
with open(os.path.join(data_path, 'schematext.sql'), 'r') as f:
    sqls = f.readlines()
print('\n'.join(sqls))
for file in os.listdir(data_path):
    if '.csv' in file:
        your_table = file.split('.')[0]
        print(f'COPY {your_table} FROM \'{os.path.join(data_path, file)}\' WITH (FORMAT CSV, DELIMITER \',\', QUOTE \'"\', ESCAPE \'\\\', HEADER);')