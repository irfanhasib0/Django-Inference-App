const express = require('express');
const mysql = require('mysql2');
const cors = require('cors');
const fs = require('fs');
const app = express();

// Add mysql database connection
const db = mysql.createPool({
  host: 'mysql_db', // the host name MYSQL_DATABASE: node_mysql
  user: 'MYSQL_USER', // database user MYSQL_USER: MYSQL_USER
  password: 'MYSQL_PASSWORD', // database user password MYSQL_PASSWORD: MYSQL_PASSWORD
  database: 'notes' // database name MYSQL_HOST_IP: mysql_db
})


const createDbQuery = "CREATE TABLE IF NOT EXISTS `user_database` (\
  `index` int(11) NOT NULL AUTO_INCREMENT,\
  `user` varchar(30) NOT NULL,\
  `topic` varchar(20) NOT NULL,\
  `section` int(11) NOT NULL,\
  `title` varchar(50) NOT NULL,\
  `content` varchar(50) NOT NULL,\
  PRIMARY KEY (`index`)\
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=latin1;";

  
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

async function createDb(Query){
sleep(10)
let result = awaitQuery(Query)
console.log('Created db ...',result)
}
createDb(createDbQuery);

// home page
app.get('/', (req, res) => {
  res.send('Hi There')
});

// get all of the books in the database
app.get('/get', (req, res) => {
  const SelectQuery = " SELECT * FROM  user_database";
  db.query(SelectQuery, (err, result) => {
    res.send(result)
  })
})

// get all of the books in the database
app.get('/getid', (req, res) => {
  const user    = req.query.user
  const topic   = req.query.topic
  const section = req.query.section
  const SelectQuery = " SELECT * FROM  user_database WHERE user = ? AND topic = ? AND section = ?";
  db.query(SelectQuery,[user,topic,section], (err, result) => {
     console.log(result)
	    fs.readFile('/app_data/files/'+String(user)+'_'+String(topic)+'_'+ String(section) + '.txt', 'utf8', (err, data) => {
		            try{result[0].content = data}
		            catch{console.log('result content not found !!! ',result,'section',section)}
		            res.send(result)
		       });
		});
	
});


// get all of the notes in the database
app.get('/getids', (req, res) => {
  const user    = req.query.user
  const topic   = req.query.topic
  const SelectQuery = " SELECT * FROM  user_database WHERE user = ? AND topic = ?";
  db.query(SelectQuery,[user,topic], (err, result) => {
                    res.send(result)
        });
});

// get all of the notes in the database
app.get('/get_users', (req, res) => {
  const SelectQuery = " SELECT DISTINCT user FROM  user_database";
  db.query(SelectQuery,[], (err, result) => {
                    res.send(result)
        });
});


// get all of the notes in the database
app.get('/get_topics', (req, res) => {
  const user    = req.query.user
  const SelectQuery = " SELECT DISTINCT topic FROM  user_database WHERE user = ?";
  db.query(SelectQuery,[user], (err, result) => {
                    res.send(result)
        });
});

// add a note section to the database
app.post("/insert", async (req, res) => {
  const user    = req.body.user;
  const topic   = req.body.topic;
  const section = req.body.section;
  const title   = req.body.title;
  const content = req.body.content;
  const InsertQuery = "INSERT INTO user_database (user, topic, section, title, content) VALUES (?, ?, ?, ?, ?)";
  const CheckQuery  = "SELECT * FROM user_database WHERE user = ? AND topic = ? AND section = ?";
  let result = await awaitQuery(CheckQuery, [user, topic, section])
  console.log('await Check query : ',user, topic, section,' Result : ', result)
  if (result.length === 0) {
     fs.writeFile('/app_data/files/'+String(user)+'_'+String(topic)+'_'+ String(section) + '.txt', String(content), error=>{console.log(error)})
     let result = await awaitQuery(InsertQuery, [user, topic ,parseInt(section), title, content])
     console.log('Insert query for : ',user, topic, section, title, content,' Result : ',result)
       
}
});
  

// delete a note section from the database
app.delete("/delete/:section", (req, res) => {
  const section = req.params.section;
  const DeleteQuery = "DELETE FROM user_database WHERE section = ?";
  db.query(DeleteQuery, section, (err, result) => {
    if (err) console.log(err);
  })
})

// update a book review
app.post("/update",async (req, res) => {
  const user    = req.body.user;
  const topic   = req.body.topic;
  const section = req.body.section;
  const title  = req.body.title;
  const content = req.body.content;
  const UpdateQuery = "UPDATE user_database SET title = ? , content = ? WHERE user = ? AND topic = ? AND section = ?";
  console.log('api',title, content,user,topic,section)
  fs.writeFile('/app_data/files/'+String(user)+'_'+String(topic)+'_'+ String(section) + '.txt', String(content) ,error=>{console.log(error)})
  let result = await awaitQuery(UpdateQuery, [title, content.slice(0,50),user,topic,section])
  console.log(result)
});

// get all of the books in the database
app.get('/get_fig/:id', (req, res) => {
  const id = req.params.id
  fs.readFile('/app_data/files/fig_'+ String(id) + '.txt', 'utf8', (err, data) => {res.send(data)});
        });


app.post("/fig_update", (req, res) => {
  const id     = req.body.id;
  const title  = req.body.title;
  const content = req.body.content;
  //const UpdateQuery = "UPDATE user_database SET title = ? , content = ? WHERE id = ?";
  console.log('api',title, content,id)
  fs.writeFile('/app_data/files/fig_'+ String(id) + '.txt', String(content) ,error=>{console.log(error)})
  //db.query(UpdateQuery, [title, content.slice(0,1000),id], (err, result) => {
  //  if (err) console.log(err)
  //})
})

app.listen('3001', () => { })
