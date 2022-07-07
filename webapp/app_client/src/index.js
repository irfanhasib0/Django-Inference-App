import React from 'react';
import ReactDOM from 'react-dom';
import TextTopic from './components/text_editor/RichTextTopic';
import DrawApp from './components/draw_editor/RichDrawArea';
import BoardApp from './components/board/BoardApp'
import "./draw/src/index.css";
import 'bootstrap/dist/css/bootstrap.min.css';
import {Container, Card, Row, Col } from 'react-bootstrap'
import Note from './note'
//<Col> <BoardApp /> </Col> 

//<Col xs={6}>
//  <App_1 id="0" />
//  </Col>
//<App_1 id="0" />
ReactDOM.render(
  <>
  <Container>
  <Row>
  <Col xs={10}>
  <TextTopic />
  </Col>
  </Row>
  </Container>
  </>,
  document.getElementById('root')
);


//<Row>  
//  <Col> <App_1 id="0" /> </Col> 
//  <Col> <DrawApp id="0"/> </Col>
//  </Row>
//ReactDOM.render(
//  <DrawApp/>,
//  document.getElementById('draw')
//);
