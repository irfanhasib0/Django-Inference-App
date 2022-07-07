import React from 'react'
import {Editor, convertFromRaw, EditorState, RichUtils} from 'draft-js';
import {Button, Container, Card, Row, Col } from 'react-bootstrap'
import TextBlock from './RichTextBlock';
import axios from 'axios'


class EditorSmp extends React.Component {
  constructor(props) {
    super(props);
    this.state = {editorState: props.state, id: props.id};
    this.onChange = editorState => this.setState({editorState});
    this.handleKeyCommand = this.handleKeyCommand.bind(this);
  }
  
  handleKeyCommand(command, editorState) {
    const newState = RichUtils.handleKeyCommand(editorState, command);

    if (newState) {
      this.onChange(newState);
      return 'handled';
    }

    return 'not-handled';
  }
  
  render() {
    return (
      <TextBlock
        id={this.state.id}
        editorState={this.state.editorState}
        onChange={this.onChange}
        handleKeyCommand={this.handleKeyCommand}
      />
    );
  }
}

class TextTopic extends React.Component {
    constructor(props){
           super(props);
           this.state = {notes:[],nextId:0}
    }
    
    
    addBlock = () =>
    {
    let notes = this.state.notes
    notes.push({'id': parseInt(this.state.nextId)})
    this.setState({notes:notes,nextId:parseInt(this.state.nextId+1)})
    //document.location.reload()
    }
    
    saveResponse(response) {
       let notes = []
       for(let res of response.data){
       notes.push({'id': parseInt(res.id)})
       this.setState({notes:notes,nextId:parseInt(res.id)+1})
      }
      }
      
    async componentDidMount() {
       const resp = await axios.get(`/api/getids`)//.then(this.saveResponse(response))
       this.saveResponse(resp)
      }
      
      
    render() {
         console.log('render notes',this.state.notes)
         let blocks = this.state.notes.map((item,index)=>{
         return (
         <>
         <EditorSmp id = {item.id}/>
         </>
         )
         });
         
         return(
         <>
         <Button style={{width : '40px', padding : '5px'}} onClick={this.addBlock}>&#10000;</Button>{' '}
         {blocks}
         </>)
        };
   }

export default TextTopic;
