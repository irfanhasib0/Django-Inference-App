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
  `id` int(11) NOT NULL AUTO_INCREMENT,\
  `book_name` varchar(50) NOT NULL,\
  `book_review` varchar(50) NOT NULL,\
  PRIMARY KEY (`id`)\
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=latin1;";
db.query(Query,(err, result) => {
    console.log(result)
  })
  
// Enable cors security headers
app.use(cors())

// add an express method to parse the POST method
app.use(express.json())
app.use(express.urlencoded({ extended: true }));

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

// add a book to the database
app.post("/insert", (req, res) => {
  const bookName = req.body.setBookName;
  const bookReview = req.body.setReview;
  const InsertQuery = "INSERT INTO books_reviews (book_name, book_review, book_content) VALUES (?, ?, ?)";
  db.query(InsertQuery, [bookName, bookReview, bookContent], (err, result) => {
    console.log(result)
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
app.put("/update/:bookId", (req, res) => {
  const bookName = req.body.setBookName;
  const bookReview = req.body.setReview;
  const bookContent = req.body.setContent;
  const bookId = req.params.bookId;
  const UpdateQuery = "UPDATE books_reviews SET book_name = ? , book_review = ? , book_content = ? WHERE id = ?";
  console.log('api',bookName,bookReview,bookId)
  db.query(UpdateQuery, [bookName,bookReview, bookContent, bookId], (err, result) => {
    if (err) console.log(err)
  })
})

app.listen('3001', () => { })
