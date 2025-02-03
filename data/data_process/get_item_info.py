import pickle


user_info = {}
with open('../movies.dat', 'r', encoding='ISO-8859-1') as f:
    for line in f:
        user_info_line = line.strip().split('::')
        user_info_line[-1] = user_info_line[-1].split('|')
        user_info[user_info_line[0]] = user_info_line

pickle.dump(user_info, open('/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/' + 'movie_info.pkl', 'wb'))

