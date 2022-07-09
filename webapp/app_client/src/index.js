import React ,{Component} from 'react';
import ReactDOM from 'react-dom';
import axios from 'axios'
import TextTopic from './components/text_editor/RichTextTopic';
import DrawApp from './components/draw_editor/RichDrawArea';
import {Container, Card, Row, Col } from 'react-bootstrap'
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
      }
      async getUsers(){
       const resp = await axios.get(`/api/get_users`)
       console.log('Users ... ', resp)
      }
      render(){
      this.getUsers();
      return (<>
      
      </>)}
      }
      
ReactDOM.render(
  <>
  <Row>
  <Users/>
  <Col xs={2}>
  <ProSidebar>
  <SidebarContent>
     <Menu iconShape="circle">
       <MenuItem icon={<FaTachometerAlt />} suffix={<span className="badge red">{'users'}</span>}>
         {'user'}
       </MenuItem>
      <MenuItem icon={<FaGem />}> {'components'}</MenuItem>
     </Menu>
  </SidebarContent>
  </ProSidebar>
  </Col>
  
  <Col xs={"auto"} style={{marginLeft : '0px'}}>
  <div background-color ="blue" style={{marginTop : '0px', marginBottom : '0px'}}>
  <h3 style={{"color":"cyan", "background-color":"white"}}> Notebook </h3>
  </div>
  <TextTopic user='user_1' topic='topic_1' />
  </Col>
  </Row>
  </>,
  document.getElementById('root')
);
