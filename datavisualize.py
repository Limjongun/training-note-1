import matplotlib.pyplot as plt

x = [1,2,3,4,5]
y = [1,4,9,16,25]

plt.plot(x, y)

plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title("basic line")
plt.show()

import matplotlib.pyplot as plt

x = [1,2,3,4,5]
y = [1,4,9,16,25]

plt.plot(x,y,color='red',linestyle='--',marker='o',linewidth=2,markersize=9) #-
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

x = [1,2,3,4,5]
y = [1,4,9,16,25]
y2 = [1,2,3,4,5]

plt.figure(figsize=(9,5))

plt.subplot(1,2,1)
plt.plot(x,y,color='green')
plt.title("Plot 1")

plt.subplot(1,2,2)
plt.plot(y,y2,color='blue')
plt.title("Plot 2")


plt.grid(True)
plt.show()


import matplotlib.pyplot as plt

categories=['A','B','C','D','E']
value=[6,7,8,9,4]
plt.bar(categories,value,color='red')
plt.show()

import matplotlib.pyplot as plt

data=[1,1,1,2,2,2,2,3,3,3,4,4,4,4,5,5,6]
plt.hist(data,bins=5,edgecolor='black')
plt.show()


import matplotlib.pyplot as plt

x= [1,2,3,4,5]
y= [2,8,4,5,2]
plt.scatter(x,y,color="blue",marker='x')
plt.show()

import matplotlib.pyplot as plt

lables=['A','B','C','D','C']
sizes=[20,30,40,50,60]
color=['gold','red','blue','green','black']
explode= (0.2,0,0,0,0)
plt.pie(sizes,labels=lables,autopct="%1.1f%%",shadow=True,explode=explode)
plt.show()



import matplotlib.pyplot as plt
import pandas as pd

sales=pd.read_csv('data.csv')
print(sales.head(5))

##plot total sales by product

total_sales_byproduct = sales.groupby('Category')['Sales'].sum()
print(total_sales_byproduct.head(5))
total_sales_byproduct.plot(kind='bar',color='teal')
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

sales = pd.read_csv('data.csv')

sales_trend = sales.groupby('Date')['Sales'].sum()

sales_trend.plot(kind='bar', color='teal')
plt.title("Sales Trend")
plt.show()