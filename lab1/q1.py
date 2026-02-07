list1= [2,7,4,1,3,6]
count=0
for i in range(len(list1)):
    for j in range(i+1,len(list1)):
                   if list1[i] + list1[j] == 10 :
                       count=count+1
print (count)
                   
                    
                   
