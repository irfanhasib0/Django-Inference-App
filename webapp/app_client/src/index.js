import React ,{Component,useState,useEffect} from 'react';
import ReactDOM from 'react-dom';
import axios from 'axios'
import TextTopic from './components/text_editor/RichTextTopic';
import DrawApp from './components/draw_editor/RichDrawArea';
import {Container, Card, Row, Col, Button } from 'react-bootstrap'
import Nav from 'react-bootstrap/Nav';
import Navbar from 'react-bootstrap/Navbar';
import NavDropdown from 'react-bootstrap/NavDropdown';

import { ProSidebar, Menu, MenuItem, SubMenu, SidebarContent, SidebarHeader, SidebarFooter, Sidebar } from './components/sidebar';
import './components/sidebar/scss/styles.scss';
import 'bootstrap/dist/css/bootstrap.min.css';
import { FaTachometerAlt, FaRegEdit, FaTrashAlt, FaGem, FaList, FaRegLaughWink, FaHeart, FaBook, FaUserCircle } from 'react-icons/fa';
import {BsJournalRichtext, BsBook, BsStack} from 'react-icons/bs';
import {AiOutlineMail} from 'react-icons/ai';

function Header(props){
        return (
	<Navbar bg='gray'>
	<Container>
		<Navbar.Brand href="#home"> <BsBook/> {props.topic} 
    <Button variant='outline-success' style={{width : '25px', height : '25px', padding : '0px', marginLeft : '5px' }} onClick={props.callbacks.renameTopic} ><FaRegEdit/></Button>
    <Button variant='outline-danger' style={{width : '25px', height : '25px', padding : '0px', marginLeft : '5px' }} onClick={props.callbacks.deleteTopic} ><FaTrashAlt/></Button>
    </Navbar.Brand>
		<Navbar.Toggle aria-controls="basic-navbar-nav" />
		<Navbar.Collapse id="basic-navbar-nav">
		  <Nav className="me-auto">
		    <Nav.Link href="#home"><FaUserCircle/> {props.user}</Nav.Link>
		    <NavDropdown title= "Editor" id="basic-nav-dropdown">
		      <NavDropdown.Item href="#action/3.1">Action</NavDropdown.Item>
		      <NavDropdown.Item href="#action/3.2">
		        Another action
		      </NavDropdown.Item>
		      <NavDropdown.Item href="#action/3.3">Something</NavDropdown.Item>
		      <NavDropdown.Divider />
		      <NavDropdown.Item href="#action/3.4">
		        Separated link
		      </NavDropdown.Item>
		    </NavDropdown>
		  </Nav>
		</Navbar.Collapse>
	  </Container>
	  </Navbar>)
 }

function Layout(props) {
const [user,setUser]           = useState('...')
const [topic,setTopic]         = useState('...')
const [topics,setTopics]       = useState(['...'])
const [userName,setUserName]   = useState('...')
const [topicName,setTopicName] = useState('...')
const [timeout,setTime]        = useState(parseInt(500))

async function getUsers(){
     let users = []
     const resp = await axios.get(`/api/get_users`)
     for (let elem of resp.data)
       {      
         users.push(elem.user)
       }
     if(users.length ===0)
     {
      createNewUser()
     }
       
     setUser(users[0])
     return users[0]
     }
     
async function getTopics(){
     let topics = []
     const resp = await axios.get(`/api/get_topics?user=${user}`)
     for (let elem of resp.data)
       {      
         topics.push(elem.topic)
       }
     if (window.localStorage.getItem('topic')){
     setTopic(window.localStorage.getItem('topic'))
     //window.localStorage.removeItem('topic')
     }
     else {
     setTopic(topics[0])
     }
     //topics.push('...')
     setTopics(topics)
     return topics[0]
     }

const renameTopic = () => {
  let name = window.prompt(topic,topic);
  let payload = {'user': user , 'topic' : topic, 'save' : 'topic', 'name' : name};
  console.log('saving data : ',payload)
  axios.post(`/api/save`, payload)
  window.localStorage.setItem('topic',name)
  document.location.reload();
  console.log('Rename : ',payload)
}

const deleteTopic = () => {
  let payload = {'user': user, 'topic' : topic , 'section' : '*'}
  if (window.confirm("Do you want to delete? ")) {
    axios.post(`/api/delete`,payload)
    document.location.reload()
  }
}

const loadMyAsyncData = () => new Promise((resolve, reject) => {
  setTimeout(() => resolve(
    getUsers()
  ), timeout)
  setTimeout(() => resolve(
    getTopics()
  ), timeout)
  setTime(1000)
  
})
 
useEffect(()=>{
loadMyAsyncData ()
})


let textBlock = (<>{'loading ...'}</>)
let topic_items = (<>
<MenuItem icon={<BsBook />}> 
{topic}
</MenuItem>
</>)

if (user!=='...' & topic!=='...'){
textBlock = (<TextTopic user={user} topic={topic} />)
topic_items = topics.map((item,index)=>{

return(<>
<MenuItem key={item} icon={<BsBook />}> 
<Button variant='outline-secondary' style={{width : 'auto', height : '25px', padding : '0px', marginBottom : '10px' , marginLeft : '10px' }} onClick={()=>{window.localStorage.setItem('topic',item); document.location.reload()}} >{'# '+item+' .'}</Button>
</MenuItem>
</>)
})
}

function sendTopicName (event){ setTopicName(event.target.value) }
function createNewTopic(){
    axios.post('/api/insert', {'user':user,'topic':topicName,'section':0,'title':'','content':''})
      .then(() => { alert('success post') })
}

function sendUserName (event){ setUserName(event.target.value) }
function createNewUser(){
    axios.post('/api/insert', {'user':'user_1','topic': 'topic_1','section':0,'title':'','content':''})
      .then(() => { alert('success post') })
    document.location.reload()
}
//<textarea style={{width : '100px', height : '30px' , marginTop : '30px'}} onChange = {sendUserName} placeholder={'user name'}></textarea>
//<Button style={{width : '25px', height : '25px', padding : '0px', marginBottom : '10px' }} onClick={createNewUser} >+</Button>
return (
  <>
  <Row>
  <Col xs={3}>
  <ProSidebar> 
  <SidebarHeader style = {{marginTop : '10px' , marginLeft : '10px'}}>
  <h5><FaUserCircle/> {user}</h5>
  </SidebarHeader>
  <SidebarContent  style = {{marginTop : '10px' , marginBottom : '250px'}}>
     <Menu iconShape="circle">
      <SubMenu defaultOpen={true} title={'Notebooks'} icon={<BsStack />} suffix={<span className="badge red">{String(topics.length)}</span>}>
      {topic_items}
      <textarea style={{width : '100px', height : '30px' , marginTop : '30px'}} onChange = {sendTopicName} placeholder={'topic name'}></textarea>
      <Button style={{width : '25px', height : '25px', padding : '0px', marginBottom : '10px' }} onClick={createNewTopic} >+</Button>
      </SubMenu>
     </Menu>
  </SidebarContent>
  <SidebarFooter>
  <div style = {{padding : '0px', marginTop : '10px' , marginBottom : '10px'}} >
 <AiOutlineMail/> {'irfanhasib.me@gmail.com'}
  </div>
  </SidebarFooter>
  </ProSidebar>
  </Col>
  <Col xs={"auto"} style={{marginLeft : '0px'}}>
  <Header user={user} topic={topic} callbacks={{'renameTopic': renameTopic, 'deleteTopic' : deleteTopic}}/>
  
  {textBlock}
  </Col>
  </Row>
  </>)
}
  
ReactDOM.render(
  <>
  <Layout/>
  </>,
  document.getElementById('root')
);
