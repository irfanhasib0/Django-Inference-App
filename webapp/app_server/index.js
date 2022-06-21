const express = require('express');
const mysql = require('mysql2');
const cors = require('cors');
const app = express();

// Add mysql database connection
const db = mysql.createPool({
  host: 'mysql_db', // the host name MYSQL_DATABASE: node_mysql
  user: 'MYSQL_USER', // database user MYSQL_USER: MYSQL_USER
  password: 'MYSQL_PASSWORD', // database user password MYSQL_PASSWORD: MYSQL_PASSWORD
  database: 'books' // database name MYSQL_HOST_IP: mysql_db
})
const Query = "CREATE TABLE IF NOT EXISTS `books_reviews` (\
  `index` int(11) NOT NULL AUTO_INCREMENT,\
  `id` int(11) NOT NULL,\
  `title` varchar(50) NOT NULL,\
  `content` varchar(50) NOT NULL,\
  PRIMARY KEY (`index`)\
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=latin1;";

  
// Enable cors security headers
app.use(cors())

// add an express method to parse the POST method
app.use(express.json())
app.use(express.urlencoded({ extended: true }));


db.query(Query,(err, result) => {
    console.log(result)
  })
  
// home page
app.get('/', (req, res) => {
  res.send('Hi There')
});

// get all of the books in the database
app.get('/get', (req, res) => {
  const SelectQuery = " SELECT * FROM  books_reviews";
  db.query(SelectQuery, (err, result) => {
    res.send(result)
  })
})

// get all of the books in the database
app.get('/getid/:id', (req, res) => {
  const id = req.params.id
  const SelectQuery = " SELECT * FROM  books_reviews WHERE id = ?";
  db.query(SelectQuery,[id], (err, result) => {
    res.send(result)
  })
})

// add a book to the database
app.post("/insert", (req, res) => {
  const id      = req.body.id;
  const title   = req.body.title;
  const content = req.body.content;
  const InsertQuery = "INSERT INTO books_reviews (id, title, content) VALUES (?, ?, ?)";
  const CheckQuery  = "SELECT * FROM books_reviews WHERE id=?"
  db.query(CheckQuery, [id], (err, result) => {
    if (result.length === 0) {
         db.query(InsertQuery, [id, title, content], (err, result) => {
         console.log(result)
         })
       }
  })
  
})

// delete a book from the database
app.delete("/delete/:bookId", (req, res) => {
  const bookId = req.params.bookId;
  const DeleteQuery = "DELETE FROM books_reviews WHERE id = ?";
  db.query(DeleteQuery, bookId, (err, result) => {
    if (err) console.log(err);
  })
})

// update a book review
app.post("/update", (req, res) => {
  const id     = req.body.id;
  const title  = req.body.title;
  const content = req.body.content;
  const UpdateQuery = "UPDATE books_reviews SET title = ? , content = ? WHERE id = ?";
  console.log('api',title, content,id)
  db.query(UpdateQuery, [title, content,id], (err, result) => {
    if (err) console.log(err)
  })
})

app.listen('3001', () => { })
