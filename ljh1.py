from fileinput import close

import DemoClass

# import math
# from tokenize import String
#
# # print("liujiahao1")
# # print('刘家豪刘、、')
# # print('liujiahao'+'刘家豪')
# # print('He'+ ' '+'said "good!"')
# # print("liujiah\n\"ao\"")
# # print("""刘家豪
# # 景行智和
# # 转正""")
# # my = 123455
# # """i = 2
# # print(i**3)"""
# # str="""@@刘家豪
# # 景行智和
# # 转正"""
# # print(str)
# # print(my)
# # # print(my)
# # print(my+3)
# # print(len("\""))
# # print(len("213"))
# # print("hello"[0])
# # print(type("null"))
# # print(type(None))
# # user_height = float(input("请输入你的身高："))
# # user_wight = float(input("请输出你的体重："))
# # user_bim = user_height/user_wight**2
# # print(user_bim)
# # if user_bim<0.025:
# #     print("你超重了")
# # elif user_bim != 0.025:
# #     print("你不是良好")
# # else:
# #     print("你很健康")
# # print(len(user_height))
# # if user_height < user_wight:
# #     print("你太胖了")
# # elif user_height > user_wight:
# #     if user_bim >= 0.025:
# #         print("你还行")
# #     else:
# #         print("你还是不行")
# # else:
# # #     print("难道你们相等")
# # print(not(user_bim<0.025))
# # print(user_height>=user_wight or user_bim < 0.025)
# # print(user_bim>=user_wight)
# # print(type(False))
# # print(type(True))
# # list = [];
# # list.append('ljh')
# # list.append("刘家豪")
# # list.append(0)
# # list.append(True)
# # list.remove("ljh")
# # print(list)
# # print(list[1] > 0)
# # t1 = ("ljh",21)
# # t2 = ("ttt",22)
# # ls = []
# # ls.append(t1)
# # ls.append(t2)
# # print(type(ls))
# # print(ls)
# # print(ls[0])
# # set = {"ljh","ljh",'ljh',"www"}
# # print(set)
# # print(type(set))
# zidian = {"ljh":21,"www":222 , "lll":33}
# dict = {("1",1):2}
# # print(zidian)
# # zidian["www"] = "555"
# # print(zidian)
# # print(type(zidian))
# # print("ljh" in zidian)
# # del zidian["lll"]
# # print(zidian)
# # print(len(zidian))
# # print(zidian.keys())
# # print(zidian.get("ljh"))
# # print(zidian.get("www"))
# # print(zidian.pop("www"))
# # print(zidian)
# # print(zidian.items())
# # for key,value in zidian.items():
# #     if key == "ljh":
# #         continue
# #     else:
# #         print(value)
# # for key,value in dict.items():
# #     if key ==("1",1):
# #        pass
# #     else:
# #         print("你好")
# range(1,10,3)
# for i in range(1,10,3):
#     if i % 2 == 0:
#         print(i)
#     else:
#         pass
# list = ["你","好","吗","我","很","好"]
# # for i in list:
# #     print(i)
# # for i in range(0,len(list)):
# #     print(list[i])
# i = 0
# while i<len(list):
#     print(list[i])
#     i=i+1
# for a  in  list:
#     print("猜猜这个字是什么:{0}".format(a))
# name = "我"
# print(f"""你好吗，{name}很好""")
# def test(t1,t2):
#     # 定义函数，模拟SUM函数，并返回值
#     s = t1 + t2
#     print(s)
#     return s
# str = test(1,2)
# # print(str)
# import statistics as st
#
# from DemoClass import Student, eat
#
# print(st.median([19.-5,36]))

# import DemoClass
# p = DemoClass.Student("刘家豪","炸鸡")
# # p.eat()
# s = DemoClass.people("ljh","蛋糕","啤酒")
# s.eat1()
# s.eat()
#
# print(s.name)

f = open("D:\\刘家豪\\Study_刘家豪\\AiAgent.md", "r", encoding="utf-8")
f1 = open(r"D:\刘家豪\Study_刘家豪\AiAgent.md", "r", encoding="utf-8")
# print(f.read(10))
# print("--------------------------------------------------------------")
# print(f.read(20))
# print(f.readline())
# print(f.readline())
# print(f1.readlines())
# f.close()
# f1.close()
# with open("D:\\刘家豪\\Study_刘家豪\\AiAgent.md", "r", encoding="utf-8") as f:
#     print(f.read())
# line = f.readline()
# i = 1
# while line:
#     print(line)
#     line = f.readline()
#     print(i)
#     i = i +1
# f.close()
# print(f1.readlines())
# l = f1.readlines()
# for line in l:
#     print(line)
# s = f1.readline()
# txt = open("D:\\刘家豪\\Study_刘家豪\\AiAgent.md", "r+", encoding="utf-8")
# txt.write("\n1111111111111111利用python追加文件内容")
# print("-----------------------------------------------")
# print(txt.read())
# txt.close()
#
#
# try:
#     s = open("qqqqq", "r", encoding="utf-8")
#     print(s.read())
#     s.close()
# except Exception as e:
#     print(e)
# else:
#     print("没有异常")
# finally:
#     print("关闭文件")
i = "1111"
assert len(i) >4
print("长度为4")