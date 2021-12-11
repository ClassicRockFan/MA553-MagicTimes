#!/bin/python
import mysql.connector
import os
from datetime import timedelta


conn = mysql.connector.connect(user='admin', password='redacted',
                              host='themeparkdata-1.ctscu93smucy.us-east-1.rds.amazonaws.com',
                              database='ParkData')
conn.autocommit = True
cursor = conn.cursor()
cursor.execute('SELECT RideId FROM ParkData.ConcernedRides WHERE ParkId IN (4,5,6,7);')
rides = cursor.fetchall()
cursor.close()
conn.close()

for ride in rides:
    print("working on ride {0}".format(ride[0]))
    conn = mysql.connector.connect(user='admin', password='redacted',
                                  host='themeparkdata-1.ctscu93smucy.us-east-1.rds.amazonaws.com',
                                  database='ParkData')

    conn.autocommit = True
    cursor = conn.cursor()
    query = ("SELECT DISTINCT\
            `CR`.`RideId` AS `RideId`, \
            `CR`.`ParkId` AS `ParkId`, \
            COALESCE(`ru`.`WaitTime`, 0) AS `WaitTime`,\
            CASE WHEN ru.WaitTime IS NULL THEN 'Borked' ELSE `ru`.`Status` END  AS `Status`,\
            CASE WHEN ru.WaitTime IS NULL THEN 0 ELSE `ru`.`IsActive` END AS `IsActive`,\
            FROM_UNIXTIME(((300 * FLOOR((UNIX_TIMESTAMP(`ru`.`CreatedTimeStamp`) / 300))) - ((4 * 60) * 60))) AS `CreatedTimeStamp` \
        FROM ParkData.ConcernedRides CR \
        JOIN ParkData.RideUpdate ru ON ru.RideId = CR.RideId \
        WHERE CR.RideId = {0};".format(ride[0]))

    cursor.execute(query)
    rows = []
    workItems = ""
    rowCount = 0
    fileNames = []
    for row in cursor:
        rows.append(row)
        if(len(rows) < 24):
            continue
        if(len(rows) > 24):
            rows = rows[1:]

        work = "INSERT INTO ParkData.FilteredParkData VALUES ({0}, {1}, '{2}'".format(rows[0][0], rows[0][1], rows[0][5])
        startTime = rows[0][5]
        dontSave = False
        for workRow in rows:
            newTime = workRow[5]
            if(newTime != startTime):

    #            print(len(rows), rows)
                dontSave = True
                break
            work = work + ", {0}, '{1}', {2}".format(workRow[2], workRow[3], workRow[4])
            startTime = newTime + timedelta(minutes = 5)
        if(dontSave):
            continue
        work = work + ");\n"
        workItems = workItems + work
        rowCount += 1
        if(rowCount % 1000 == 0):
            print("Processed {0} rows!".format(rowCount))
            f = open("{0}.sql".format(rowCount), 'w+')
            f.write(workItems)
            f.close()
            fileNames.append("{0}.sql".format(rowCount))
            workItems = ""

    print("Processed {0} rows!".format(rowCount))
    f = open("{0}.sql".format(rowCount), 'w+')
    f.write(workItems)
    f.close()
    fileNames.append("{0}.sql".format(rowCount))
    workItems = ""

    cursor.close()
    conn.close() 
    for file in fileNames:
        conn = mysql.connector.connect(user='admin', password='redacted',
                                  host='themeparkdata-1.ctscu93smucy.us-east-1.rds.amazonaws.com',
                                  database='ParkData')

        conn.autocommit = True
        cursor = conn.cursor()
        cursor = conn.cursor() 
        print(file)
        f = open(file)
        text = f.read()
        f.close()
        cursor.execute(text)
        cursor.close()  
        conn.close()
        os.remove(file) 


    print("completed parsing data for {0} rows of ride {1}".format(rowCount, rows[0][0]))