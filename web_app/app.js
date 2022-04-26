const express = require('express')
const app = express()
const expressLayouts = require("express-ejs-layouts")
const path = require('path')
const fs = require('fs');
const shell = require('shelljs')

app.set('view engine', 'ejs');
app.use(express.urlencoded({extended: true }))
app.set('views', path.join(__dirname, '/views'))
app.set('layout', 'layouts/layout')
app.use(expressLayouts)
app.use(express.static( path.join(__dirname, '/public')))




app.post('/all', async(req, res)=>{
    let m = new Date();
    let dateString = m.getUTCFullYear() +"/"+ (m.getUTCMonth()+1) +"/"+ m.getUTCDate() + " " + m.getUTCHours() + ":" + m.getUTCMinutes() + ":" + m.getUTCSeconds();
    let reload = req.body['force']
    let domain_name = parse_domain(req, res)
    let raw_data = undefined

    try{
        let user_agent = req.body['UA']
        console.log(`[${dateString}] Using user agent: ${user_agent}`)
    }catch(error){
        
    }


    let cached = ['fitcrack.fit.vutbr.cz', 'bio-senpai.ovi.moe', 'danbooru.donmai.us', 'fit.vut.cz']

    if(cached.includes(domain_name)){
	    reload = "off"
    }
    console.log(`[${dateString}] resolver started for ${domain_name}`)

    const path = `./sites/${domain_name}.json`


    if(!fs.existsSync(path)){
        shell.exec(`cd domain_evaluation/src && python3 init.py ${domain_name} --silent && mv ${domain_name}.json ../../sites`)
    }else{
        if(reload == 'on'){
            shell.exec(`rm ${path}`);
            shell.exec(`cd domain_evaluation/src && python3 init.py ${domain_name} --silent && mv ${domain_name}.json ../../sites`)
        }else{
            console.log(`using cache for [${domain_name}]`)
            await sleep(5000);
        }
    }


    try{
        if(!fs.existsSync(path)){
            return res.send({"Status": 500, "Message": "Internal server error"})
        }else{
            let domain_data = JSON.parse(fs.readFileSync(path));
            // logger setup
            try{
                let now = new Date();
                let nowString = now.getUTCFullYear() +"/"+ (now.getUTCMonth()+1) +"/"+ now.getUTCDate() + " " + now.getUTCHours() + ":" + now.getUTCMinutes() + ":" + now.getUTCSeconds();
                let fin = parseFloat(((new Date() - m)/1000)).toFixed(2);
    
                let code = domain_data['combined']
                let acc = domain_data['accuracy']
                console.log(`[${nowString}] ${fin} s for ${domain_name}, ${code} with ${acc} accuracy`)
            }catch(error){
                console.log("LOGGGER ERROR")	
                console.log(error)	
            }

            return res.send({"Status": 200, "Data": domain_data})
        }
    }catch(error){
        return res.send({"Status": 500, "Message": "Internal server error"})
    }

})


app.post('/feedback', async(req, res)=>{
    console.log(req.body)
})

app.post('/lexical', async(req, res)=>{
    return api(req, res, '--lexical')
})

app.post('/data', async(req, res)=>{
    return api(req, res, '--data_based')
})


app.post('/svm', async(req, res)=>{
    return api(req, res, '--svm')
})


app.post('/get-raw-data', async(req, res)=>{
    let domain_name = parse_domain(req, res)


})

app.get('/*', (req, res)=>{
    res.render('index')
})


let port = 4444
app.listen(port, () => {console.log(`Listening on ${port}`)})


// --- functions for domain name validation --- //

function domainValidate(domain) {
    

    if (/^[www.]*[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9](?:\.[a-zA-Z]{2,})+$/.test(domain)) {
        return true
    }else{
        console.log("Not valid domain name")
        return false;
    }
}


function parse_domain(req,res){
    let domain_name = req.body['domain_name']
    if(domain_name == ''){
        return res.send({"Status": 401, "Message": "Wrong request, expecting domain name"})
    }


    if(!domainValidate(domain_name)){
        return res.send({"Status": 401, "Message": "Invalid domain name"})
    }

    return domain_name
}


function api(req, res, mode){
    let domain_name = parse_domain(req,res)

    const path = `./${domain_name}.json`
    shell.exec(`cd domain_evaluation/src && python3 init.py ${domain_name} ${mode} && mv ${domain_name}.json ../..`)

    try{
        if(!fs.existsSync(path)){
            return res.send({"Status": 500, "Message": "Internal server error"})
        }else{
            let domain_data = JSON.parse(fs.readFileSync(path));
            shell.exec(`rm ${path}`)
            return res.send({"Status": 200, "Data": domain_data})
        }
    }catch(error){
        return res.send({"Status": 500, "Message": "Internal server error"})
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
