s="hippopotamus"
mostrep= ''
freq=0
for i in s:
    count=0
    for j in s:
        if i==j:
            count = count+1
        if count>freq:
            freq=count
            mostrep=i
print("Highest repeating character : ",mostrep)
print("Frequency count : ",freq)
