# DanteMode

# ‼️ Big rewrite with less bad code coming soon


# update 2 im rewriting this for a third time but i havent even published the second one
# (i have a love hate relationship with asyncio.run_coroutine_threadsafe())
rewrite is technically done but I need to fix up the code so it relys on paths relative to the repository instead of my system 

(In private development for ~6 months) Dante, my discord bot. Open source project in development and not optimized for other filesystems, install and setup script in development

Use with your own caution, a lot of development needs to be done and this is not plug and play at the moment.



https://user-images.githubusercontent.com/88869424/137430048-d1e715f1-0d6e-438a-b7af-04d3b95347c9.mp4



What I need to do:
- test universal install
- basic instructions on how to manually set up

Whats been done:
- Make a setup script
- Untested universal install
- Source code
- Actual bot release
- Optimized ML for speed and realism (on gtx 1060 6 gb with cuda, 45 seconds for decent image)
- New model

What you need to run it:
- decent cpu
- gtx 1060, nvidia equivelant or better
- cuda support
- python
- a will to untangle a mess in the case it doesnt work

Install instructions:
1. Do 'git clone https://github.com/Ednaordinary/DanteMode'
2. cd into DanteMode and run install.sh, follow the instructions
3. When install has finished, run start.sh (run start everytime you start dante, but not install)
4. Enjoy dante
