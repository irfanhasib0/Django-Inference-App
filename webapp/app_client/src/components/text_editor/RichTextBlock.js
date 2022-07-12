
//https://www.section.io/engineering-education/build-and-dockerize-a-full-stack-react-app-with-nodejs-and-nginx/
//https://github.com/mosesreigns/Build-and-Dockerize-a-Full-stack-React-app-with-Node.js-MySQL-and-Nginx-for-reverse-proxy
//https://github.com/sstur/react-rte
//https://dev.to/pccprint/10-react-rich-text-editors-1hh5#:~:text=react%2Dweb%2Deditor&text=The%20React%20Web%20editor%20is,styling%20component's%20color%20and%20text.
/* eslint-disable no-unused-expressions */
/* eslint-disable no-restricted-globals */

import React, { Component, useRef } from 'react';
import './RichTextArea.css';
import axios from 'axios';
import { Button, IconButton, Container, Card} from 'react-bootstrap'
import { Editor } from 'react-draft-wysiwyg-local-a';
import 'react-draft-wysiwyg-local-a/dist/react-draft-wysiwyg.css';
import { convertFromRaw, AtomicBlock, AtomicBlockUtils, convertToRaw, EditorState, Modifier } from 'draft-js';
import CanvasDraw from "react-canvas-draw";
import rough from "roughjs/bundled/rough.esm";
import DrawApp from './RichDrawRender';
import PropTypes from 'prop-types';
import { FaRegEdit, FaSave, FaTrashAlt } from 'react-icons/fa';

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
        user:props.user,
        topic:props.topic,
        section: props.section,
        title: '',
        content: '',
        editorState: EditorState.createEmpty(),
        entid : -1,
        toolbarHidden: true,
        bottomToolbarHidden: true,
        clicked:false,
        readOnly:false
      }
      this.setDomEditorRef = ref => this.domEditor = ref;
  }
  // *********************************************************************//
  loadContent = (content) =>{
  console.log('loading ...')
  if (content) {
    console.log(content)
    content = JSON.parse(content)
    this.setState({editorState : EditorState.createWithContent(convertFromRaw(content))})
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
    //this.showText(rawContentState)
    this.setState({content : JSON.stringify(rawContentState)})
    
  }
  
  // *********************************************************************//
  saveContent = () => {
  //window.localStorage.setItem('content', JSON.stringify(rawContentState));
  console.log("Saving ...")
  axios.post(`/api/update`, this.state).then(() => { alert('success post') })
  this.setState({clicked:true})
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
      console.log('Get id',`/api/getid?user=${this.state.user}&topic=${this.state.topic}&section=${this.state.section}`)
        axios.get(`/api/getid?user=${this.state.user}&topic=${this.state.topic}&section=${this.state.section}`)
      .then((response) => {
        console.log('Get id : ',response.data)
        this.loadContent(response.data[0].content);
      })
      if (this.state.section===0){this.domEditor.focusEditor()}
      
  }
  //*************************************************************************
  getPayload = () => {
  let payload = {  'user'    : this.state.user,
                   'topic'   : this.state.topic,
                   'section' : this.state.section,
                   'title'   : this.state.title,
                   'content' : this.state.content}
   return payload
  }
  
  // *********************************************************************//
  submit = () => {
    let payload = this.getPayload()
    console.log('Submit request with paylad',payload)
    axios.post('/api/insert', payload).then(() => { alert('success post') })
    console.log(this.state)
  }
  
  // *********************************************************************//
  rename = () => {
  
  }
  
  
  // *********************************************************************//
  update = (id) => {
    let payload = this.getPayload()
    console.log('updating with data : ',payload)
    axios.post(`/api/update`, payload)
    document.location.reload();
  }
  
  // *********************************************************************//
  delete = (id) => {
    if (confirm("Do you want to delete? ")) {
      axios.delete(`/api/delete/${id}`)
      this.setState({clicked:true})
      document.location.reload()
    }
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
  
  // ***************************************************************************
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
  
  onEditCallback = () => {
  //this.domEditor.focusEditor()
  //this.setState({toolbarHidden : false})
  //this.setState({bottomToolbarHidden : false})
  }
  
  onFocusCallback= () => {
  this.setState({toolbarHidden : false})
  this.setState({bottomToolbarHidden : false})
  }
  
  onBlurCallback= () => {
  setTimeout(()=>{
  this.setState({toolbarHidden : true})
  if (this.state.clicked==false)
      {
	  this.setState({
	  bottomToolbarHidden : true,
	  })
      }
  else{
  this.setState({
          bottomToolbarHidden : false,
	  clicked : false,
	  })
  }
  },200)
  }
  
  render() {
    //<Button variant="outlined-primary" style={{marginLeft : '10px', width : 'auto', padding : '2px'  }} onClick={this.onEditCallback()}>{<FaRegEdit/>}</Button>{' '}
    return (
            <>
            <Card style={{height: 'auto', marginTop : '10px', marginBottom : '10px' }}>
            <Editor spellCheck={true} readOnly={this.state.readOnly} ref={this.setDomEditorRef}  toolbarHidden={this.state.toolbarHidden} editorClassName={this.state.title}  id="text" rows="10" cols="30" class="note" 
            onEditorStateChange={this.handleEditorChange}
            editorState={this.state.editorState}
            blockRendererFn={this.blockRendererFn}
            toolbarCustomButtons={[<CustomOption />]}
            onFocus={this.onFocusCallback}
            onBlur={this.onBlurCallback}
            toolbar={{
            options: ['inline', 'blockType', 'fontSize', 'fontFamily', 'list', 'textAlign', 'colorPicker', 'link', 'emoji', 'image', 'history'],
            list: { inDropdown: true },
            textAlign: { inDropdown: true },
            link: { inDropdown: true },
            history: { inDropdown: true },
            image: {previewImage: true}
            }} >
            </Editor>
            </Card>
            {(this.state.bottomToolbarHidden) ? (<></>) : 
		    (<div style={{height: '40px' }}>
		    <Button variant="outlined-success" style={{width : 'auto', padding : '2px' }} onClick={() => { this.saveContent() }}>{<FaSave/>}</Button>{' '}
		    <Button variant="outlined-danger" style={{width : 'auto', padding : '2px' }}  onClick={() => { this.delete(this.state.id) }}>{<FaTrashAlt/>}</Button>{' '}
		    </div>)
            }
            </>
            
    );
  }
}
export default TextBlock;
