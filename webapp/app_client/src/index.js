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
import { FaTachometerAlt, FaRegEdit, FaGem, FaList, FaRegLaughWink, FaHeart, FaBook, FaUserCircle } from 'react-icons/fa';
import {BsJournalRichtext, BsBook, BsStack} from 'react-icons/bs';
import {AiOutlineMail} from 'react-icons/ai';

function Header(props){
        return (
	<Navbar bg='gray'>
	<Container>
		<Navbar.Brand href="#home"> <BsBook/> {props.topic} </Navbar.Brand>
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
const [user,setUser]   = useState('...')
const [topic,setTopic] = useState('...')
const [topics,setTopics] = useState(['...'])
const [topicName,setTopicName] = useState('...')
const [timeout,setTime] = useState(parseInt(500))

async function getUsers(){
     let users = []
     const resp = await axios.get(`/api/get_users`)
     for (let elem of resp.data)
       {      
         users.push(elem.user)
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
     }
     else {
     setTopic(topics[0])
     }
     topics.push('...')
     setTopics(topics)
     return topics[0]
     }
     
const loadMyAsyncData = () => new Promise((resolve, reject) => {
  setTimeout(() => resolve(
    getUsers()
  ), timeout)
  setTimeout(() => resolve(
    getTopics()
  ), timeout)
  setTime(500)
  
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
  <Header user={user} topic={topic}/>
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
