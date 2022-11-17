import {Container, Card, Row, Col, Button } from 'react-bootstrap'
import 'bootstrap/dist/css/bootstrap.min.css';
import {useState, useRef, useEffect} from 'react'
import logo from './logo.svg';
import './App.css';
import bg1 from './img/bg1.jpg'
import {BsJournalRichtext, BsBook, BsStack, BsVectorPen,BsGem,BsLinkedin,BsFillTerminalFill} from 'react-icons/bs';
import {AiOutlineMail, AiOutlineLogin, AiOutlineLogout, AiOutlineCaretUp, AiOutlineCopy} from 'react-icons/ai';
import axios from 'axios';

import Dropdown from 'react-bootstrap/Dropdown';
import ButtonGroup from 'react-bootstrap/ButtonGroup';
import VisBlock from './VisBlock'

function DropdownBtn(props) {
  return (
    <Dropdown as={ButtonGroup}>
      <Dropdown.Toggle variant="success" id="dropdown-basic">
        {props.name}
      </Dropdown.Toggle>

      <Dropdown.Menu>
        {props.items}
        </Dropdown.Menu>
    </Dropdown>
  );
}

//export default BasicExample;


function App() {
   const fref = useRef('null')
   
   useEffect(()=>{
   
   
  })
    
  const [layerOutput,setLayerOutput] = useState('')
  const [imageOutput,setImageOutput] = useState('')
  const [imageDir,setImageDir]       = useState('')
  
  const modelNames = ['resnetv2_50','mobilenetv2']
  const imageDirs  = ['car','train']
  const imageNames = ['1.jpg','2.jpg','3.jpg']
  
  async function get_layer(layer_name){
      const resp = await axios.get('http://localhost:9001/layer/'+layer_name+'/')
      setLayerOutput(resp.data['src'])
  }

  async function get_image(image_name){
      setImageOutput(image_name)
      const resp = await axios.get('http://localhost:9001/image/'+imageDir+'/'+imageOutput)
      setImageOutput(resp.data['src'])
  }

  const [elems,setElem] = useState(<div>  </div>)
  async function get_model(model) {
     const resp = await axios.get('http://localhost:9001/'+model+'/')//.then(response => { console.log(response.data) });
     let elems = []
     let key = 0
     for (let elem of resp.data){
         elems.push(<div> <Button variant='outline-info' style={{height:'40px', marginTop : '10px'}} onClick={()=>get_layer(elem)} key={key} > {elem} </Button> </div>)
         key = key + 1
     }
      
      setElem(elems)
    }
  
  //backgroundImage : `url(${bg1})`
  const getModelMenuItem=(name)=>{
  return (<Dropdown.Item> 
  <Button variant='outline-success' style={{width : '25px', height : '25px', padding : '0px', marginLeft : '5px' }} onClick={()=>{get_model(name)}} ><BsBook/></Button>{name}
  </Dropdown.Item>)
  }
  
  const getImgDirItem=(name)=>{
  return (<Dropdown.Item> 
  <Button variant='outline-success' style={{width : '25px', height : '25px', padding : '0px', marginLeft : '5px' }} onClick={()=>{setImageDir(name)}} ><BsBook/></Button>{name}
  </Dropdown.Item>)
  }
  
  const getImgNameItem=(name)=>{
  return (<Dropdown.Item> 
  <Button variant='outline-success' style={{width : '25px', height : '25px', padding : '0px', marginLeft : '5px' }} onClick={()=>{get_image(name)}} ><BsBook/></Button>{name}
  </Dropdown.Item>)
  }
  
  const modelMenueItems = modelNames.map(getModelMenuItem)
  const imgDirItems = imageDirs.map(getImgDirItem)
  const imgNameItems = imageNames.map(getImgNameItem)
  return (
    <>
      
      
      <Row>
      
      <Col xs={6}>
      <VisBlock/>
      </Col>
     
      <Col xs={6} style={{backgroundColor : 'ffffff' }}>
      {}
      </Col>
     
      </Row> 
      </>
  );
}

export default App;
