import React from 'react';
import ReactDOM from 'react-dom';
import App_1 from './RichTextArea';
import DrawApp from './draw/src/RichDrawArea';
import "./draw/src/index.css";
import 'bootstrap/dist/css/bootstrap.min.css';
import {Container, Card, Row, Col } from 'react-bootstrap'

ReactDOM.render(
  <Row>  
  <Col> <App_1 id="0" /> </Col> 
  <Col> <DrawApp/> </Col>
  </Row>,
  document.getElementById('text')
);

//ReactDOM.render(
//  <DrawApp/>,
//  document.getElementById('draw')
//);
