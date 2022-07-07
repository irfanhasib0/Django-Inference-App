
//https://www.section.io/engineering-education/build-and-dockerize-a-full-stack-react-app-with-nodejs-and-nginx/
//https://github.com/mosesreigns/Build-and-Dockerize-a-Full-stack-React-app-with-Node.js-MySQL-and-Nginx-for-reverse-proxy
//https://github.com/sstur/react-rte
//https://dev.to/pccprint/10-react-rich-text-editors-1hh5#:~:text=react%2Dweb%2Deditor&text=The%20React%20Web%20editor%20is,styling%20component's%20color%20and%20text.
/* eslint-disable no-unused-expressions */
/* eslint-disable no-restricted-globals */

import React, { Component, useRef } from 'react';
import './App.css';
import axios from 'axios';
import { Button, Container, Card, Row, Col } from 'react-bootstrap'
//import {RichTextEditor} from 'components/editor/src'
import { Editor } from 'react-draft-wysiwyg';
import { convertFromRaw, AtomicBlockUtils, convertToRaw, EditorState } from 'draft-js';
import 'react-draft-wysiwyg/dist/react-draft-wysiwyg.css';
import CanvasDraw from "react-canvas-draw";
import rough from "roughjs/bundled/rough.esm";
//import DrawApp from './draw/src/RichDrawAreaLine';

const CanvasBlock = ({ contentState, block, blockProps: { onSave, content }, ...rest }) => {
  const canvas = useRef(null);
  return <div
    className="canvas-container"
    onMouseUp={() => {
      const entity = block.getEntityAt(0);
      onSave(contentState.replaceEntityData(entity, { content: canvas.current.getSaveData() }))
    }}
  >
    <CanvasDraw
      canvasWidth={350}
      canvasHeight={300}
      brushRadious={3}
      saveData={content}
      ref={canvas}
    />
  </div>
};

class App_1 extends Component {
  constructor(props) {
    super(props)
      this.state = {
        id: props.id,
        title: '',
        content: '',
        editorState: ''
        //EditorState.createEmpty()
      }
  }
  // *********************************************************************//
  loadContent = (content) =>{
  //const content = window.localStorage.getItem('content');
  console.log('loading ...')
  if (content) {
    console.log(content)
    content = JSON.parse(content)
    this.setState({editorState : EditorState.createWithContent(convertFromRaw(content))})
    //this.handle
  } else {
    this.setState({editorState : EditorState.createEmpty()})
  }
  }
  
  
  // *********************************************************************//
  handleEditorChange = (editorState) => {
    
    this.setState({
      editorState,
    });
    
    const contentState    =  editorState.getCurrentContent();
    //console.log(contentState.getEntityMap())
    const rawContentState =  convertToRaw(contentState);
    //this.saveContent(rawContentState)
    this.showText(rawContentState)
    this.setState({content : JSON.stringify(rawContentState)})
    //this.setState({content : 'test content abc'})
    
  }
  
  // *********************************************************************//
  saveContent = () => {
  //window.localStorage.setItem('content', JSON.stringify(rawContentState));
  console.log(this.state)
  axios.post(`/api/update`, this.state).then(() => { alert('success post') })
  //document.location.reload();
  }
  
  // *********************************************************************//
  showText(rawContentState)
  {
  console.log('text ...',convertFromRaw(rawContentState).getPlainText())
  }
  
  // *********************************************************************//
  componentDidMount() {
      this.submit();
        axios.get(`/api/getid/${this.state.id}`)
      .then((response) => {
        this.loadContent(response.data[0].content);
      })
  }
  
  // *********************************************************************//
  submit = () => {
    axios.post('/api/insert', this.state)
      .then(() => { alert('success post') })
    console.log(this.state)
    //document.location.reload();
  }
  
  // *********************************************************************//
  rename = () => {
  
  }
  
  // *********************************************************************//
  delete = (id) => {
    if (confirm("Do you want to delete? ")) {
      axios.delete(`/api/delete/${id}`)
      document.location.reload()
    }
  }
  
  // *********************************************************************//
  update = (id) => {
    console.log('update',id,this.state)
    axios.post(`/api/update`, this.state)
    document.location.reload();
  }
   
  // *********************************************************************// 
  textAreaForm = () => {
  return (
    <div className="form-group">
      <label htmlFor="text">Basic textarea</label>
      <textarea
        className="form-control"
        id="text"
        rows="5"
      />
    </div>
  );
  };
  
  insertCanvas = () => {
    const editorState = this.state.editorState;
    let content = editorState.getCurrentContent();

    content = content.createEntity(
      'CANVAS',
      'IMMUTABLE',
      { content: '' }
    )

    const entityKey = content.getLastCreatedEntityKey();

    this.setState({
      editorState: AtomicBlockUtils.insertAtomicBlock(
        editorState,
        entityKey,
        ' ',
      ),
    });
  };

  saveCanvas = (content) => {
    this.setState({
      editorState: EditorState.push(
        this.state.editorState,
        content
      )
    });
  };

  blockRendererFn = block => {
    const editorState = this.state.editorState;
    if (editorState){
	    console.log(editorState)
	    const content = editorState.getCurrentContent();

	    if (block.getType() === 'atomic') {
	      const entityKey = block.getEntityAt(0);
	      const entity = content.getEntity(entityKey);
	      console.log('entity',entity)
	      const entityData = entity.getData() || { content: '' }
	      if (entity != null && entity.getType() === 'CANVAS') {
		return {
		  component: CanvasBlock,
		  props: {
		    onSave: this.saveCanvas,
		    ...entityData
		  }
		}
	      }
	    }
	 }
    };


  //<div class="row">
  //<input name='setBookName' placeholder={this.state.title} onChange={this.handleChange} />
  //<label htmlFor="text"> {this.state.title} </label>
  //<Button className='m-2' onClick={() => { this.renameit(0) }}>Rename</Button>
  //</div>
  //
  render() {
    
    //let editorState = this.state.editorState;
    return (
            <React.Fragment>
            <Editor editorClassName={this.state.title}  id="text" rows="5" cols="30" class="note" 
            onEditorStateChange={this.handleEditorChange}
            editorState={this.state.editorState}
            blockRendererFn={this.blockRendererFn} >
            </Editor>
            <button onClick={this.insertCanvas}>&#10000;</button>
            <Button className='m-2' onClick={() => { this.saveContent() }}>Save</Button>
            <Button onClick={() => { this.delete(0) }}>Delete</Button>
            </React.Fragment>
    );
  }
}
export default App_1;
