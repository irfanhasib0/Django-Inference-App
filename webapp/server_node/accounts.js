const main = require('./init');
const app = main.app;
const db  = main.db;
const awaitQuery = main.awaitQuery;

const Query = "CREATE TABLE IF NOT EXISTS `users` (\
  `index` int(11) NOT NULL AUTO_INCREMENT,\
  `user` varchar(30) NOT NULL,\
  `password` varchar(20) NOT NULL,\
  `token` varchar(11) NOT NULL,\
  PRIMARY KEY (`index`)\
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=latin1;";

const createAdmin = "INSERT INTO users (user,password,token) VALUES (? , ? , ?)";
  
async function createDB(){
let result = await awaitQuery(Query)
console.log('Created db : ',result)
}
createDB();

async function createAdminUser(){
  let result = await awaitQuery(createAdmin,['admin','1234','1234'])
  console.log('Created admin user : ',result)
  }
createAdminUser();


// update a book review
app.post("/get_token",async (req, res) => {
  const user        = req.body.user;
  const password    = req.body.password;
  console.log(user)
  const SelectQuery = " SELECT token  FROM  users WHERE user = ? AND password = ?";
  db.query(SelectQuery,[user,password],(err,result)=>{
    console.log(result)
    res.send(result)
  })
    
});

// update a book review
app.post("/get_user",async (req, res) => {
  const token       = req.body.token;
  const SelectQuery = " SELECT user  FROM  users WHERE token = ?";
  db.query(SelectQuery,[token],(err,result)=>{
    console.log(result)
    res.send(result)
  })
    
});

// update a book review
app.post("/set_token",async (req, res) => {
  const user        = req.body.user;
  const password    = req.body.password;
  const token       = req.body.section;

  const InsertQuery = " INSERT token = ? INTO  user_database WHERE user = ? AND password = ?";
  let result = awaitQuery(InsertQuery,[token,user,password])
  res.send(result)
    
});

exports.app = app;
exports.awaitQuery = awaitQuery;
exports.db = db;