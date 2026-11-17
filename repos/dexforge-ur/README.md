# dexforge-ur

Scaffolded OEM repository for Universal Robots.

Layout:
- `rust/` - no_std Rust crate for the UR driver
- `python/` - Python bindings crate for `dexforge_ur`

Standalone OEM package layout for in-workspace builds.

To run with URSim:

docker pull universalrobots/ursim_e-series
docker run -d --name ursim `
  -e ROBOT_MODEL=UR5 `
  -p 5900:5900 `
  -p 6080:6080 `
  -p 29999:29999 `
  -p 30001:30001 `
  -p 30002:30002 `
  -p 30003:30003 `
  -p 30004:30004 `
  universalrobots/ursim_e-series


  URSim boots with the simulated arm powered off. Either click the power button in PolyScope (http://localhost:6080/vnc.html) or send power on\n followed by brake release\n to 127.0.0.1:29999 over TCP before starting your RTDE connection.

Host IP: Both native Windows processes and WSL2 processes connect to 127.0.0.1.