import {Container, Card, Row, Col, Button } from 'react-bootstrap'
import 'bootstrap/dist/css/bootstrap.min.css';
import {useState, useRef, useEffect} from 'react'
import logo from './logo.svg';
import './App.css';
import bg1 from './img/bg1.jpg'
import {BsJournalRichtext, BsBook, BsStack, BsVectorPen,BsGem,BsLinkedin,BsFillTerminalFill} from 'react-icons/bs';
import {AiOutlineMail, AiOutlineLogin, AiOutlineLogout, AiOutlineCaretUp, AiOutlineCopy} from 'react-icons/ai';
import axios from 'axios';

function App() {
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
  
  async function get_layer(layer_name){
      const resp = await axios.get('http://localhost:9001/layer/'+layer_name+'/')
      setLayerOutput(resp.data['src'])
  }

  async function get_image(image_name){
      const resp = await axios.get('http://localhost:9001/image/'+image_name)
      setImageOutput(resp.data['src'])
  }

  const [elems,setElem] = useState(<div> '...' </div>)
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
  //get_layer()
  //console.log(layerOutput)
  //
  //{layerOutput}
  //backgroundImage : `url(${bg1})`
  return (
    <>
      
      <Row>
      <Col xs={2} style={{backgroundColor : 'gray' }}>
      {elems}
      </Col>
      
      <Col xs={9}>
      <div style={{marginLeft : "80px", marginTop : "10px"}}>
      <Button variant='outline-success' style={{width : '25px', height : '25px', padding : '0px', marginLeft : '5px' }} onClick={()=>{get_model('resnetv2_50')}} ><BsBook/></Button>{'ResNetV2_50'}
      
      <Button variant='outline-success' style={{width : '25px', height : '25px', padding : '0px', marginLeft : '5px' }} onClick={()=>{get_image('car/2.jpg')}} ><BsBook/></Button>{'ResNetV2_50'}
      </div>

      <div>
      <img style={{display:'block'}} src={'data:image/jpg;base64,' + imageOutput}/>
      <img style={{}} src={'data:image/png;base64,' + layerOutput}/>
      </div>
      
      <iframe ref={fref} src={""} scrolling="no" width={"100%"} height={"10000 px"}></iframe>
      </Col>
      </Row> 
      </>
  );
}

export default App;
