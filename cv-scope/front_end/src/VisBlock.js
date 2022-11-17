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


function VisBlock() {
   const fref = useRef('null')
   
   useEffect(()=>{
   const iframe = fref.current
   
   iframe.onload = function()
        {
          iframe.style.height = 
          iframe.contentWindow.document.body.scrollHeight + 'px';
          iframe.style.width  = 
          iframe.contentWindow.document.body.scrollWidth  + 'px';
              
        }
        
   
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
  //<DropdownBtn items = {modelMenueItems} name='Models'/>
  return (
    <>
      
      <Row>
      <div style={{marginLeft:'10px', marginTop:'10px'}} >
      <DropdownBtn items = {modelMenueItems} name='Models'/>
      </div>
      <Col xs={4} style={{backgroundColor : '#eeeeee' }}>
      <div style={{marginLeft:'10px', marginTop:'10px'}} >
      
      {elems}
      </div>
      </Col>
      
      <Col xs={8}>
      <div style={{marginLeft : "0px", marginTop : "10px"}}>
      <DropdownBtn style={{}} items = {imgDirItems} name='Image Class'/>
      {' '}
      <DropdownBtn items = {imgNameItems} name='Image No'/>
      </div>

      <div style={{marginLeft:'0px'}}>
      
      <img style={{display:'block', marginLeft:'50px', marginTop: '50px'}} src={'data:image/jpg;base64,' + imageOutput}/>
      <img style={{width:'100%',marginLeft:'0px'}} src={'data:image/png;base64,' + layerOutput}/>
      
      </div>
      <iframe ref={fref}/>
      </Col>
      </Row> 
      </>
  );
}

export default VisBlock;
