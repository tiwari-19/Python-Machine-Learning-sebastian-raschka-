#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 09:46:10 2017

@author: ashish
"""

import sqlite3
import os

conn = sqlite3.connect('Reviews.sqlite')
c = conn.cursor()

c.execute('DROP TABLE IF EXISTS Review_db')
c.execute('CREATE TABLE IF NOT EXISTS Review_db (Review TEXT, Sentiment INT, Date TEXT);')
ex1 = 'I love this movie'
ex2 = 'I hate this movie'

c.execute("INSERT INTO Review_db (Review, Sentiment, Date) VALUES (?, ?, DATETIME('now')) ", (ex1, 1, ))
c.execute("INSERT INTO Review_db (Review, Sentiment, Date) VALUES (?, ?, DATETIME('now')) ", (ex2, 0, ))
conn.commit()


c.execute("SELECT * FROM Review_db")
results = c.fetchall()
for row in results:
       print row
conn.close()