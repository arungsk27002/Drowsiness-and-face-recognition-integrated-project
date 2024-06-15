class Student:
    collegename="mar athanasius college of engineering"
    naam="arun das"#class attributes
    def __init__(self,fullname,amarks):
        self.naam=fullname
        self.marks = amarks
    def hello(self):
        print("welcome student",self.naam)
    def getmarks(self):
        print("marks is ",self.marks)
    @staticmethod
    def hello():
        print("hello")
s1 =Student("arun",97)
s2=Student("das",98)
print(s1.naam ,s1.marks,s1.collegename)
print(s2.naam,s2.marks)
print(Student.collegename)
s1.hello()
s2.getmarks()