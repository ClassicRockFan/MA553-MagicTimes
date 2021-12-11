#!/bin/python
import mysql.connector
import os
from datetime import timedelta
import csv


conn = mysql.connector.connect(user='admin', password='#T0ny44ss267_',
                              host='themeparkdata-1.ctscu93smucy.us-east-1.rds.amazonaws.com',
                              database='ParkData')
conn.autocommit = True
cursor = conn.cursor()
cursor.execute("CALL GetUniversalData();")
print(cursor)

# Write result to file.
with open("universalData.csv", 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    column_names = []
    for i in cursor.description:
        column_names.append(i[0])
    csvwriter.writerow(column_names)
    for row in cursor:
        csvwriter.writerow(row)

cursor.close()
conn.close()
