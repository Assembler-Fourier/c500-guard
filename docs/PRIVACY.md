# Local data and retention

This source package contains no camera credentials, active source configuration, camera pictures, calibration templates, recordings, sighting crops, vehicle database, authorization notes or model weights. Example addresses use documentation or loopback ranges. Test geometry has no camera image or geographical reference and is marked non-deployable.

A running installation does create sensitive local data. Camera and alignment configuration stay in the ignored `config/` tree or application state directory. Incident clips and optional paired clean-frame archives share incident retention. Vehicle sightings and the appearance report have their own configured retention; the vehicle-report default is 12 hours. Backups and manually exported evidence need separate handling: deleting active records does not imply that copies elsewhere were erased.

Enable vehicle sightings and the report only with appropriate authorization. The report worker looks for a local `AUTHORISATION*.txt` or `AUTHORIZATION*.txt` note in its directory or parent. Do not commit that note or put credentials in it. The presence-camera feature is separate and disabled in the generic example.

The dashboard and report are local. Report images are embedded in the locally generated report and are not uploaded by the application. Appearance suggestions are not plate recognition or verified identity. An operator still needs to assess ambiguous direction, night colour and possible matches.

Before publication, review the actual proposed Git tree as well as `.gitignore`: ignored files can still be added explicitly. Exclude active `config`, runtime state, backups, screenshots, databases, generated reports, local authorization notes and credentials. The repository's test runner uses a temporary checkout and fake services; use that runner rather than a live installation for development checks.
