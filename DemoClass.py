class Student:
    def __init__(self, name,foot):
        self.name = name
        self.foot = foot
    def eat(self):
        print(f"{self.name}来当第一个布娃娃")
        print(f"今天要吃{self.foot}")



# people = Student("People")
# print(people.name)
# people.eat()
# print(f"这个人叫{people.name}")

class people(Student):
    # def __init__(self, name,foot):
    #     super().__init__(name)
    #     self.foot = foot
    def __init__(self,name,foot,bear):
        super().__init__(name,foot)
        self.bear = bear

    def eat1(self):
        print(f"{self.name}准备吃{self.foot}"+f"{self.bear}")

