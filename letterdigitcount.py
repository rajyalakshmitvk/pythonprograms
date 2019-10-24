name=input('Enter string')
dcount=0
lcount=0
ocount=0
for i in name:
    if(i.isdigit()):
        dcount+=1
    elif(i.isalpha()):
        lcount+=1
    else:
        ocount+=1
print('Letter Count=',lcount)
print('Digit Count=',dcount)
print('Others Count=',ocount)