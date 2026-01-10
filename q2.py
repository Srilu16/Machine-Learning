list1=[5,3,8,1,0,4]
if len(list1)<3 :
    print ("Range determination not possible")
else :
    max = list1[0]
    for i in range (len(list1)):
        if list1[i]>max:
            max=list1[i]
    min=list1[0]
    for j in range (len(list1)):
        if list1[j]<min:
            min=list1[j]
    range=max-min
print ("Range is ",range)
