import React, {useState, useEffect} from 'react'
import {Editor, convertFromRaw, EditorState, RichUtils} from 'draft-js';
import {Button, Container, Card, Row, Col } from 'react-bootstrap'
import TextBlock from './RichTextBlock';
import axios from 'axios'


class EditorSmp extends React.Component {
  constructor(props) {
    super(props);
    this.state = {editorState: props.state, user: props.user, topic: props.topic, section: props.section, isFocused: props.isFocused};
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
        user={this.state.user}
        topic={this.state.topic}
        section={this.state.section}
        editorState={this.state.editorState}
        onChange={this.onChange}
        handleKeyCommand={this.handleKeyCommand}
        isFocused={this.state.isFocused}
      />
    );
  }
}

class TextTopic extends React.Component {
    constructor(props){
           super(props);
           this.state = {notes:[],nextId:0, user:props.user,topic:props.topic, isFocused:false}
    }
    
    
    addBlock = () =>
    {
    let notes = this.state.notes
    notes.push({'user': this.state.user, 'topic': this.state.topic, 'section': parseInt(this.state.nextId)})
    this.setState({notes:notes,nextId:parseInt(this.state.nextId)+1})
    document.location.reload()
    }
    
    saveResponse(response) {
       let notes = []
       for(let res of response.data){
       //if (res.user !== '...') {
       notes.push({'user':res.user, 'topic': res.topic, 'section': parseInt(res.section)})
       this.setState({notes:notes,nextId:parseInt(res.section)+1})
       }
      //}
      }
    
    
    async componentDidMount() {
       const resp = await axios.get(`/api/getids?user=${this.state.user}&topic=${this.state.topic}`)//.then(this.saveResponse(response))
       this.saveResponse(resp)
      }
    
    render() {
         const setFocused = () =>{
          this.setState({isFocused:true})
         }
         
         console.log('Render notes',this.state.user, this.state.topic,this.state.notes)
         let blocks = this.state.notes.map((item,index)=>{
         console.log('Item id : ',item.section)
         return (
         <>
         <EditorSmp user = {this.state.user} topic = {item.topic} section = {item.section} isFocused = {[this.state.isFocused, setFocused]}/>
         </>
         )
         });
         
         return(
         <>
         {blocks}
         <Button style={{width : '60px', padding : '5px', marginLeft : '10px', marginBottom : '10px'}} onClick={this.addBlock}> + </Button>{' '}
         </>)
        };
   }


function _TextTopic(props){
    const [notes,setNotes]   = useState([])
    let [nextId,setNextId]   = useState(0)
    let user     = props.user
    let topic    = props.topic
    const [timeout,setTime] = useState(500)
    function addBlock(){
    notes.push({'user': user, 'topic': topic, 'section': parseInt(nextId)})
    setNextId(parseInt(nextId)+1)
    document.location.reload()
    }
    
    async function getNotes () {
       let _notes = []
       const response = await axios.get(`/api/getids?user=${user}&topic=${topic}`)
       for(let res of response.data){
       _notes.push({'user':res.user, 'topic': res.topic, 'section': parseInt(res.section)})
       setNextId(parseInt(res.section)+1)
       }
       setNotes(_notes)
      }
   //getNotes();
   const loadMyAsyncData = () => new Promise((resolve, reject) => {
    setTimeout(() => resolve(
    getNotes()
    ), timeout)
    })
   //loadMyAsyncData();
   useEffect(() =>{
       loadMyAsyncData();
       setTime(5000)
      });
    
    const [isFocused,setFocused] = useState(false)     
    //console.log('Render notes',user,notes)
    let blocks = notes.map((item,index)=>{
       //console.log('Item id : ',item.section)
       return (
       <>
       <EditorSmp user = {user} topic = {item.topic} section = {item.section} isFocused = {[isFocused, setFocused]}/>
       </>
       )
       });
         
    return(
       <>
       {blocks}
       <Button style={{width : '60px', padding : '5px', marginLeft : '10px', marginBottom : '10px'}} onClick={addBlock}> + </Button>{' '}
       </>)
    };

export default TextTopic;
