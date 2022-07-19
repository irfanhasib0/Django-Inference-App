const express = require('express');
const mysql = require('mysql2');
const cors = require('cors');
const app = express();

// Add mysql database connection
const db = mysql.createPool({
    host: 'mysql_db', // the host name MYSQL_DATABASE: node_mysql
    user: 'MYSQL_USER', // database user MYSQL_USER: MYSQL_USER
    password: 'MYSQL_PASSWORD', // database user password MYSQL_PASSWORD: MYSQL_PASSWORD
    database: 'notes' // database name MYSQL_HOST_IP: mysql_db
})

// Enable cors security headers
app.use(cors())

// add an express method to parse the POST method
app.use(express.json())
app.use(express.urlencoded({ extended: true }));


async function awaitQuery(query, params) {
    inTransaction = true;
    return new Promise((resolve, reject) => {
      if ( typeof params === `undefined` ) {
        db.query(query, (err, result) => {
          if ( err ){
              reject(err);
            }
          else {
            resolve(result);
          }
        });
      } else {
        db.query(query, params, (err, result) => {
          if ( err ){
              reject(err);
            }
          else {
            resolve(result);
          }
        });
      }
    });
  }
  
function sleep(sec){
for(let i=0;i<sec*10000;i++){
    for(let j=0;j<sec*10000;j++)
    {
    }
    }
}

exports.app = app;
exports.awaitQuery = awaitQuery;
exports.db = db;
