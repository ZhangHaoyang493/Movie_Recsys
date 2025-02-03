import pickle


user_info = {}
with open('../users.dat', 'r') as f:
    for line in f:
        user_info_line = line.strip().split('::')
        user_info[user_info_line[0]] = user_info_line

pickle.dump(user_info, open('/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/' + 'user_info.pkl', 'wb'))

