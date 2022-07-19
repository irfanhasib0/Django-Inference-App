const fs = require('fs');
const main = require('./accounts');
const app = main.app;
const db  = main.db;
const awaitQuery = main.awaitQuery;

const Query = "CREATE TABLE IF NOT EXISTS `user_database` (\
    `index` int(11) NOT NULL AUTO_INCREMENT,\
    `user` varchar(30) NOT NULL,\
    `topic` varchar(20) NOT NULL,\
    `section` int(11) NOT NULL,\
    `title` varchar(50) NOT NULL,\
    `content` varchar(50) NOT NULL,\
    PRIMARY KEY (`index`)\
  ) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=latin1;";
  
async function createDB(){
    let result = await awaitQuery(Query)
    console.log('Created db : ',result)
    }
createDB();
      
// home page
app.get('/', (req, res) => {
    res.send('...')
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
app.post("/delete",async (req, res) => {
    const user    = req.body.user;
    const topic   = req.body.topic;
    const section = req.body.section;
    if (section==='*'){
    const DeleteQuery = "DELETE FROM user_database WHERE user = ? AND topic = ?";
    let result = await awaitQuery(DeleteQuery, [user,topic,section])
    console.log('Delete : ',result)
    }
    else{
    const DeleteQuery = "DELETE FROM user_database WHERE user = ? AND topic = ? AND section = ?";
    let result = await awaitQuery(DeleteQuery, [user,topic,section])
    console.log('Delete : ',result)
    }
    
})

// update a book review
app.post("/save",async (req, res) => {
    const user     = req.body.user;
    const topic    = req.body.topic;
    const section  = req.body.section;
    const title    = req.body.title;
    const content  = req.body.content;
    const save     = req.body.save;
    const name     = req.body.name; // title to rename user / topic
    
    if (save==='user'){
    const UpdateQuery = "UPDATE user_database SET topic = ? WHERE user = ? AND topic = ?";
    console.log('api',title, content,user,topic,section)
    fs.rename('/app_data/files/'+String(user)+'_'+String(name)+'_'+ String(section) + '.txt', '/app_data/files/'+String(user)+'_'+String(topic)+'_'+ String(section) + '.txt')
    let result = await awaitQuery(UpdateQuery, [name,user,topic])
    console.log(result)
    }

    else if (save==='topic'){
    const UpdateQuery = "UPDATE user_database SET topic = ? WHERE user = ? AND topic = ?";
    console.log('api',user,topic,save,name)
    fs.rename('/app_data/files/'+String(user)+'_'+String(name)+'_'+ String(section) + '.txt', '/app_data/files/'+String(user)+'_'+String(topic)+'_'+ String(section) + '.txt',()=>{console.log('rename done !')})
    let result = await awaitQuery(UpdateQuery, [name,user,topic])
    console.log(result)
    }

    else {
    const UpdateQuery = "UPDATE user_database SET title = ? , content = ? WHERE user = ? AND topic = ? AND section = ?";
    console.log('api',title, content,user,topic,section)
    fs.writeFile('/app_data/files/'+String(user)+'_'+String(topic)+'_'+ String(section) + '.txt', String(content) ,error=>{console.log(error)})
    let result = await awaitQuery(UpdateQuery, [title, content.slice(0,50),user,topic,section])
    console.log(result)
    }
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

