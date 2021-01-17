from RS_main import *
infile = open('test_user.txt', 'r')
lines = []
for line in infile:
    lines.append(line)
for line in lines:
    number = int(line)
    my_dict=main_fun(number)
    with open('output.csv', 'w') as f:
        for key in my_dict.keys():
            f.write("%s,%s\n"%(key,my_dict[key]))