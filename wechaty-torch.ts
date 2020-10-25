import { Message, Wechaty } from 'wechaty'
import { ScanStatus } from 'wechaty-puppet'
import { PuppetPadplus } from 'wechaty-puppet-padplus'
import QrcodeTerminal from 'qrcode-terminal'
var request = require('request')


const token = 'YOUR_TOKEN_HERE'

const puppet = new PuppetPadplus({
  token,
})


const name  = 'wechaty-torch'

const bot = new Wechaty({
  name: name,
  puppet,
})

bot.on('scan', (qrcode, status) => {
  if (status === ScanStatus.Waiting) {
    QrcodeTerminal.generate(qrcode, {
      small: true
    })
  }
})


bot.on('login'  , user => console.info('Bot', `bot login: ${user}`))

bot.on('message', async (msg: Message) => {
    if (msg.type() !== Message.Type.Audio) {
      return
    }
    const file = await msg.toFileBox();
    const bsimg = file.toBase64();
    var formData = {
      bsimg: bsimg,
    }
    try{
      request.post({url:'http://127.0.0.1:8000/message', formData: formData}, function (error:any, response:any, body:any) {  
          if (error) {
              console.log('Error :', error)
              return
          }
          console.log(' Body :', body)
          var response = JSON.parse(body)
          if(body.length > 0){
            const pred: string = response['pred']
            const other = response['other']
            msg.say(pred+'\n'+other)
          }
      })
    }catch(e){
      console.log(e)
    }
  })
    
      
bot.start().catch(async e => {
  console.info('Bot', 'init() fail:' + e)
  await bot.stop()
  process.exit(-1)
})