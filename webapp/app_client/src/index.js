import React ,{Component,useState,useEffect} from 'react';
import ReactDOM from 'react-dom';
import axios from 'axios'
import TextTopic from './components/text_editor/RichTextTopic';
import DrawApp from './components/draw_editor/RichDrawArea';
import {Container, Card, Row, Col, Button } from 'react-bootstrap'

import Note from './note'
import { ProSidebar, Menu, MenuItem, SubMenu, SidebarContent, Sidebar } from './components/sidebar';
import './components/sidebar/scss/styles.scss';
import './draw/src/index.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import { FaTachometerAlt, FaGem, FaList, FaGithub, FaRegLaughWink, FaHeart } from 'react-icons/fa';




//async componentDidMount() {
//       const resp = await axios.get(`/api/get_topics?user=${this.state.user}&topic=${this.state.topic}`)//.then(this.saveResponse(response))
//       this.saveResponse(resp)
//      }
class Users extends Component {
      constructor(props){
      super(props)
      this.state = {users : [], setUser : props.setUser}
      }
      
      async componentDidMount(){
       const resp = await axios.get(`/api/get_users`)
       let users = []
       for (let elem of resp.data)
       {      
         users.push(elem.user)
       }
       this.setState({users:users})
       this.state.setUser(users[0])
      }
      render(){
      return (<>
      <p> {this.state.users} </p>
      </>)}
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
  setTime(2000)
  
})
 
useEffect(()=>{
loadMyAsyncData ()
})


let topic_items = (<>
<MenuItem icon={<FaGem />}> 
{topic}
</MenuItem>
</>)

let textBlock = (<>{'loading ...'}</>)

if (user!=='...' & topic!=='...'){
textBlock = (<TextTopic user={user} topic={topic} />)
topic_items = topics.map((item,index)=>{
//    <button>{item}</button>
return(<>
<MenuItem key={item} icon={<FaGem />}> 
<Button variant='outline-secondary' style={{width : 'auto', height : 'auto', padding : '0px', marginBottom : '10px' }} onClick={()=>{window.localStorage.setItem('topic',item); document.location.reload()}} >{item}</Button>
</MenuItem>
</>)
})

}

function sendTopicName (event){ setTopicName(event.target.value) }
function createNewTopic(){
    axios.post('/api/insert', {'user':user,'topic':topicName,'section':0,'title':'','content':''})
      .then(() => { alert('success post') })
}
//
return (
  <>
  <Row>
  <Col xs={2}>
  <ProSidebar> 
  <p>{user}</p>
  <SidebarContent>
     <Menu iconShape="circle">
       <SubMenu defaultOpen={true} title={'Topics'} icon={<FaTachometerAlt />} suffix={<span className="badge red">{String(topics.length)}</span>}>
      {topic_items}
      <textarea style={{width : '100px', height : '30px' , marginTop : '30px'}} onChange = {sendTopicName} placeholder={'topic name'}></textarea>
      <Button style={{width : '25px', height : '25px', padding : '0px', marginBottom : '10px' }} onClick={createNewTopic} >+</Button>
      
      </SubMenu>
      
     </Menu>
  </SidebarContent>
  </ProSidebar>
  </Col>
  <Col xs={"auto"} style={{marginLeft : '0px'}}>
  <div background-color ="blue" style={{marginTop : '0px', marginBottom : '0px'}}>
  <h3 style={{"color":"cyan", "background-color":"white"}}> Notebook {user} </h3>
  </div>
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
