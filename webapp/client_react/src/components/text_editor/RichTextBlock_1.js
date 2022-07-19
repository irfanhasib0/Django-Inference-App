
//https://www.section.io/engineering-education/build-and-dockerize-a-full-stack-react-app-with-nodejs-and-nginx/
//https://github.com/mosesreigns/Build-and-Dockerize-a-Full-stack-React-app-with-Node.js-MySQL-and-Nginx-for-reverse-proxy
//https://github.com/sstur/react-rte
//https://dev.to/pccprint/10-react-rich-text-editors-1hh5#:~:text=react%2Dweb%2Deditor&text=The%20React%20Web%20editor%20is,styling%20component's%20color%20and%20text.
/* eslint-disable no-unused-expressions */
/* eslint-disable no-restricted-globals */

import React, { Component, useRef } from 'react';
import './RichTextArea.css';
import axios from 'axios';
import { Button, Container, Card} from 'react-bootstrap'
//import {RichTextEditor} from 'components/editor/src'
//import { Editor } from 'react-draft-wysiwyg';
import { convertFromRaw, AtomicBlock, AtomicBlockUtils, convertToRaw, Editor, EditorState, Modifier } from 'draft-js';
import 'react-draft-wysiwyg/dist/react-draft-wysiwyg.css';
import CanvasDraw from "react-canvas-draw";
import rough from "roughjs/bundled/rough.esm";
import DrawApp from './RichDrawRender';
import PropTypes from 'prop-types';
import Toolbar from "./toolbar";



class CustomOption extends Component {
  static propTypes = {
    onChange: PropTypes.func,
    editorState: PropTypes.object,
  };

  addStar: Function = (): void => {
    const { editorState, onChange } = this.props;
    const contentState = Modifier.replaceText(
      editorState.getCurrentContent(),
      editorState.getSelection(),
      '⭐',
      editorState.getCurrentInlineStyle(),
    );
    onChange(EditorState.push(editorState, contentState, 'insert-characters'));
  };

  render() {
    return (
      <div onClick={this.addStar}>⭐</div>
    );
  }
}

const CanvasBlock = ({ contentState, block, blockProps: { id, entid, onSave }}) => {
  //className="canvas-container"
  //onMouseUp={() => {
  //  const entity = block.getEntityAt(0);
  //  const canvas = document.getElementById('render_'+String(entid));
  //  //onSave(contentState.replaceEntityData(entity, { content: canvas }))
  //}}
  return <div>
    <DrawApp
      id={id}
      entid={entid}
    />
  </div>
};


const CanvasBlockk = (props) => {
  
  return <div>
    <DrawApp
      id={props.id}
      entid={props.entid}
    />
  </div>
};


class TextBlock extends Component {
  constructor(props) {
    super(props)
      this.state = {
        id: props.id,
        title: '',
        content: '',
        editorState: EditorState.createEmpty(),
        entid : -1
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
    const rawContentState =  convertToRaw(contentState);
    this.showText(rawContentState)
    this.setState({content : JSON.stringify(rawContentState)})
    
  }
  
  // *********************************************************************//
  saveContent = () => {
  //window.localStorage.setItem('content', JSON.stringify(rawContentState));
  console.log(this.state)
  axios.post(`/api/update`, this.state).then(() => { alert('success post') })
  document.location.reload();
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
    let newid = this.state.entid + 1
    this.setState({
      editorState: AtomicBlockUtils.insertAtomicBlock(
        editorState,
        entityKey,
        ' ',
      ),
      entid : newid
    });
    console.log('eid...',this.state.entid)
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
	      const entityData = entity.getData() || { content: '' }
	      console.log('entity',entity)
	      if (entity != null && entity.getType() === 'CANVAS') {
		return {
		  component: CanvasBlock,
		  props: {
		    id : 0,
		    entid: this.state.entid,
		  }
		}
	      }
	    }
	 }
    };

  render() {
    
    const editorState = this.state.editorState;
    
    return (
            <>
            <Card>
            <Toolbar
            editorState={editorState}
            updateEditorState={this.handleEditorChange.bind(this)}
            />
            <Editor editorClassName={this.state.title}  id="text" rows="10" cols="30" class="note" 
            onChange={this.handleEditorChange.bind(this)}
            editorState={this.state.editorState}
            blockRendererFn={this.blockRendererFn}
            >
            </Editor>
            </Card>
            <div style={{height: '40px' }}>
            <Button style={{width : '40px', padding : '5px'}} onClick={this.insertCanvas}>&#10000;</Button>{' '}
            <Button style={{width : '100px', padding : '5px' }} onClick={() => { this.saveContent() }}>Save</Button>{' '}
            <Button style={{width : '100px', padding : '5px'}} onClick={() => { this.delete(this.state.id) }}>Delete</Button>{' '}
            </div>
            </>
            
    );
  }
}
export default TextBlock;
