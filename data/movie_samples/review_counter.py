#  count the number of reviews in a particular file
# rev_file = open('snippetaa.txt')
# new_lines = 0
# print(rev_file.readline())
# c = 0
# # s = rev_file.read()
# # s.count('\n')
# for line in rev_file:
# 	try:
# 		print("hi")
# 		if line == "":
# 			c += 1
# 	except:
# 		continue

# print("reviews", c)
line_num = 0
review_count = 0
sumary = "review/summary:"
sum_len = len(sumary)
with open('snippetab.txt', 'r', encoding='latin-1') as f:
	for line in f.readlines():
	        if line.startswith(sumary):
	            # str=line[sum_len:]
	            # f_new.write(str)
	            review_count += 1
	# f_new.close()
	# f.close()
	print(review_count)

#     while True:
#         try:
#             line = next(f)
#             if(line == "\n"):
#             	count += 1
#             # code
#         except StopIteration:
#             break
#         # except UnicodeDecodeError:
#             # code
#             # print("line_num: ", line_num)
#         # line_num +=  1
# print(count)
# # print("tot line_num: ", line_num)

